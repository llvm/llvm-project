//===-- Implementation of dlopen ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "dlopen.h"

#include "src/__support/CPP/mutex.h"
#include "src/__support/OSUtil/linux/syscall_wrappers/newfstatat.h"
#include "src/__support/common.h"
#include "src/__support/elf/bind.h"
#include "src/__support/elf/load_module.h"
#include "src/__support/elf/passive_abi.h"
#include "src/__support/elf/search.h"
#include "src/__support/macros/config.h"
#include "src/dlfcn/dl_internal.h"
#include "src/unistd/environ.h"

#include <fcntl.h>
#include <sys/stat.h>
namespace LIBC_NAMESPACE_DECL {

namespace {

bool same_name(const char *a, const char *b) {
  if (a == nullptr || b == nullptr)
    return false;
  for (; *a == *b; ++a, ++b)
    if (*a == '\0')
      return true;
  return false;
}

constexpr size_t PATH_BUFFER = 4096;

// The path with the "." and ".." components and the doubled slashes taken
// out, without asking the filesystem. Returns false when it does not fit.
bool normalise(const char *path, char *out) {
  size_t length = 0;
  const bool absolute = path[0] == '/';
  if (absolute)
    out[length++] = '/';
  const size_t root = length;
  for (const char *p = path; *p != '\0';) {
    while (*p == '/')
      ++p;
    const char *start = p;
    while (*p != '\0' && *p != '/')
      ++p;
    const size_t size = static_cast<size_t>(p - start);
    if (size == 0 || (size == 1 && start[0] == '.'))
      continue;
    if (size == 2 && start[0] == '.' && start[1] == '.') {
      // Back over the last component, unless it is one that stays.
      size_t end = length;
      if (end > root && out[end - 1] == '/')
        --end;
      size_t begin = end;
      while (begin > root && out[begin - 1] != '/')
        --begin;
      if (begin < end &&
          !(end - begin == 2 && out[begin] == '.' && out[begin + 1] == '.')) {
        length = begin;
        continue;
      }
      if (absolute)
        continue;
    }
    if (length + size + 2 >= PATH_BUFFER)
      return false;
    if (length > root && out[length - 1] != '/')
      out[length++] = '/';
    for (size_t i = 0; i < size; ++i)
      out[length++] = start[i];
  }
  out[length] = '\0';
  return true;
}

const char *last_component(const char *path) {
  const char *last = path;
  for (const char *p = path; *p != '\0'; ++p)
    if (*p == '/')
      last = p + 1;
  return last;
}

// Whether two paths name the same file: the same once their "." and ".."
// are taken out, or, when they end in the same name, the same device and
// inode, which is what a symbolic link comes to. An object reached by two
// paths has to be one object, as it holds the program's state of whatever it
// defines.
bool same_file(const char *known, const char *wanted) {
  if (known == nullptr || wanted == nullptr)
    return false;
  bool slash = false;
  for (const char *p = wanted; *p != '\0'; ++p)
    slash = slash || *p == '/';
  if (!slash || known[0] != '/')
    return false;
  char a[PATH_BUFFER];
  char b[PATH_BUFFER];
  if (!normalise(known, a) || !normalise(wanted, b))
    return false;
  if (same_name(a, b))
    return true;
  if (!same_name(last_component(a), last_component(b)))
    return false;
  struct stat one;
  struct stat two;
  if (!linux_syscalls::newfstatat(AT_FDCWD, a, &one, 0) ||
      !linux_syscalls::newfstatat(AT_FDCWD, b, &two, 0))
    return false;
  return one.st_dev == two.st_dev && one.st_ino == two.st_ino;
}

// Whether the set already holds something of this name, by either the path
// it was opened under, the file that path comes to, or the name it says it
// goes by.
bool already_loaded(const elf::ModuleSet &set, const char *name) {
  for (size_t i = 0; i < set.count; ++i)
    if (same_name(set.modules[i].name(), name) ||
        same_file(set.modules[i].name(), name) ||
        same_name(set.modules[i].soname(), name))
      return true;
  return false;
}

void run_init_array(const elf::Module &module) {
  auto array = module.dynamic().address(DT_INIT_ARRAY);
  auto size = module.dynamic().value(DT_INIT_ARRAYSZ);
  if (!array || !size)
    return;
  auto **functions = reinterpret_cast<void (**)()>(*array);
  for (size_t i = 0; i < *size / sizeof(void *); ++i)
    if (functions[i] != nullptr)
      functions[i]();
}

} // anonymous namespace

LLVM_LIBC_FUNCTION(void *, dlopen, (const char *path, int)) {
  // The run path that applies is the one belonging to whatever called, which
  // is found from where the call is returning to. Taken before anything else,
  // so nothing stands between this and the caller.
  const ElfW(Addr) caller =
      reinterpret_cast<ElfW(Addr)>(__builtin_return_address(0));

  elf::ModuleSet &set = elf::loaded_modules();
  cpp::lock_guard lock(dl::dl_mutex);

  // Without a loader there is no module set to add to, and nothing the caller
  // opened could be linked against the program.
  if (!set.linked) {
    dl::set_error("dlopen is only available in a dynamically linked process");
    return nullptr;
  }
  // A null path asks for a handle to the main program.
  if (path == nullptr)
    return dl::handle_for(0);

  for (size_t i = 0; i < set.count; ++i) {
    if (same_name(set.modules[i].name(), path) ||
        same_file(set.modules[i].name(), path) ||
        same_name(set.modules[i].soname(), path)) {
      ++set.references[i];
      return dl::handle_for(i);
    }
  }

  if (set.count >= set.capacity) {
    dl::set_error("too many shared objects are open");
    return nullptr;
  }

  const char *library_path =
      elf::library_path_from(reinterpret_cast<char **>(environ));

  const elf::Module *caller_module = nullptr;
  size_t caller_index = 0;
  for (size_t i = 0; i < set.count; ++i) {
    if (set.modules[i].contains(caller)) {
      caller_module = &set.modules[i];
      caller_index = i;
      break;
    }
  }

  // The executable, which is always the first module, takes its run path
  // against the file it really is, as it did when the process started.
  char executable_path[elf::MAX_EXECUTABLE_PATH];
  const char *caller_origin = nullptr;
  if (caller_module != nullptr && caller_index == 0)
    caller_origin = elf::executable_origin(
        caller_module->name(), executable_path, sizeof(executable_path));

  auto loaded = elf::find_and_load(path, caller_module, caller_origin,
                                   library_path, set.page_size);
  if (!loaded.has_value()) {
    dl::set_error(path, "cannot open shared object file");
    return nullptr;
  }

  const size_t first = set.count;
  set.modules[first] = loaded.value().module;
  set.mappings[first] = loaded.value().mapping;
  // Nothing loaded now has a place in the block each thread already has, so
  // there is no offset from a thread pointer to record. Each thread takes a
  // block of its own for it, the first time it asks.
  set.tls_offsets[first] = 0;
  set.references[first] = 1;
  elf::set_module_count(set, first + 1);

  // What the object was linked against has to be there too, and so does what
  // that was linked against. Anything already loaded is left where it is and
  // simply referred to again.
  bool failed = false;
  const char *missing = nullptr;
  for (size_t i = first; i < set.count && !failed; ++i) {
    const elf::Module requester = set.modules[i];
    requester.for_each_needed([&](const char *name) {
      if (failed || already_loaded(set, name))
        return;
      if (set.count >= set.capacity) {
        failed = true;
        missing = name;
        return;
      }
      auto dependency = elf::find_and_load(name, &requester, nullptr,
                                           library_path, set.page_size);
      if (!dependency.has_value()) {
        failed = true;
        missing = name;
        return;
      }
      const size_t at = set.count;
      set.modules[at] = dependency.value().module;
      set.mappings[at] = dependency.value().mapping;
      set.tls_offsets[at] = 0;
      set.references[at] = 1;
      elf::set_module_count(set, at + 1);
    });
  }

  // Bind against everything now loaded, the new objects included, so they can
  // refer to themselves and to each other as well as to what was there.
  elf::BindResult result;
  if (!failed) {
    elf::SearchOrder order(set.modules, set.count);
    order.set_tls_offsets(set.tls_offsets);
    for (size_t i = first; i < set.count; ++i) {
      const elf::BindResult one = elf::bind_module(set.modules[i], i, order);
      result.bound += one.bound;
      result.unresolved += one.unresolved;
    }
  }

  if (failed || result.unresolved != 0) {
    for (size_t i = set.count; i > first; --i)
      elf::unmap_module(set.mappings[i - 1]);
    elf::set_module_count(set, first);
    ++set.generation;
    dl::set_error(failed ? "cannot open a shared object it was linked against"
                         : "shared object has undefined symbols");
    (void)missing;
    return nullptr;
  }

  // A thread may still hold a block for a module that used to be at one of
  // these indices. Stamping them with the generation this reaches is what
  // tells it that block is not for what is there now.
  ++set.generation;
  for (size_t i = first; i < set.count; ++i)
    set.mappings[i].generation = set.generation;

  // Dependencies are initialised before what needs them, so the new modules
  // are run backwards.
  for (size_t i = set.count; i > first; --i)
    run_init_array(set.modules[i - 1]);
  return dl::handle_for(first);
}

} // namespace LIBC_NAMESPACE_DECL
