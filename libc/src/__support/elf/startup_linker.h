//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// The startup dynamic linker
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_ELF_STARTUP_LINKER_H
#define LLVM_LIBC_SRC___SUPPORT_ELF_STARTUP_LINKER_H

#include "hdr/elf_macros.h"
#include "hdr/elf_proxy.h"
#include "hdr/link_macros.h"
#include "hdr/types/size_t.h"
#include "src/__support/OSUtil/io.h"
#include "src/__support/elf/bind.h"
#include "src/__support/elf/load_module.h"
#include "src/__support/elf/module.h"
#include "src/__support/elf/startup_stack.h"
#include "src/__support/elf/thread_pointer.h"
#include "src/__support/elf/tls_block.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace elf {

// How many objects one process may load at startup.
constexpr size_t MAX_STARTUP_MODULES = 64;
// Where a bare SONAME is looked for. There is no ld.so.cache: musl manages
// without one and a glibc compatible cache format would buy nothing here.
constexpr const char *DEFAULT_SEARCH_PATHS[] = {"/lib", "/usr/lib"};

class StartupLinker {
public:
  LIBC_INLINE StartupLinker(size_t page_size, char **envp)
      : page_size_(page_size), library_path_(find_library_path(envp)) {}

  // Loads the executable's dependency graph, binds it, sets up thread local
  // storage and runs the initialisers. Returns false with a message already
  // written to stderr if anything failed.
  LIBC_INLINE bool link(const StartupStack &stack,
                        const ExecutableImage &image) {
    // The executable is already mapped; describe it where the kernel put it.
    const ElfW(Addr) bias = executable_bias(image);
    modules_[count_++] = Module(image.phdrs, image.phnum, bias, "");

    // The kernel maps the executable but applies nothing. A position
    // independent one still has relative relocations of its own, and without
    // them even its init array bounds point at the wrong addresses.
    modules_[0].relocations().apply_relative(bias);

    // Breadth first, so a dependency named earlier is loaded earlier, which
    // is the order symbol lookup then walks.
    for (size_t i = 0; i < count_; ++i) {
      bool failed = false;
      const Module requester = modules_[i];
      requester.for_each_needed([&](const char *name) {
        if (failed || already_loaded(name))
          return;
        if (!load_dependency(requester, name))
          failed = true;
      });
      if (failed)
        return false;
    }

    // Thread local storage is laid out before binding, because a relocation
    // against a thread local stores an offset from the thread pointer and
    // that offset is not known until the layout is fixed. Only the layout
    // though: what goes in the blocks is copied once binding is done.
    if (!setup_tls())
      return false;

    SearchOrder order(modules_, count_);
    order.set_tls_offsets(tls_offsets_);
    for (size_t i = 0; i < count_; ++i) {
      BindResult result = bind_module(modules_[i], i, order);
      if (result.unresolved != 0) {
        report("symbol not found while linking ");
        report(modules_[i].name());
        if (result.missing != nullptr) {
          report(": ");
          report(result.missing);
        }
        report("\n");
        return false;
      }
    }

    // The initialisation images are part of the modules' data, so they are
    // only what they should be once the modules have been relocated. A
    // thread local holding the address of anything is written by one.
    copy_tls_images(modules_, count_, tls_block_.thread_pointer, tls_offsets_);

    // Only once everything is bound may any of it run.
    if (!install_thread_pointer())
      return false;

    run_initialisers();
    return true;
  }

  LIBC_INLINE size_t module_count() const { return count_; }

private:
  // The kernel maps the executable but does not say what bias it used, so it
  // is recovered the same way as for any other module.
  LIBC_INLINE static ElfW(Addr) executable_bias(const ExecutableImage &image) {
    for (ElfW(Half) i = 0; i < image.phnum; ++i)
      if (image.phdrs[i].p_type == PT_PHDR)
        return reinterpret_cast<ElfW(Addr)>(image.phdrs) -
               image.phdrs[i].p_vaddr;
    return 0;
  }

  LIBC_INLINE bool already_loaded(const char *name) const {
    for (size_t i = 0; i < count_; ++i) {
      const char *soname = modules_[i].soname();
      if (soname != nullptr && equal(soname, name))
        return true;
    }
    return false;
  }

  LIBC_INLINE static bool equal(const char *a, const char *b) {
    for (; *a == *b; ++a, ++b)
      if (*a == '\0')
        return true;
    return false;
  }

  // LD_LIBRARY_PATH, without the value being copied anywhere: the environment
  // outlives the loader.
  LIBC_INLINE static const char *find_library_path(char **envp) {
    if (envp == nullptr)
      return nullptr;
    const char key[] = "LD_LIBRARY_PATH=";
    for (char **e = envp; *e != nullptr; ++e) {
      const char *p = *e;
      size_t i = 0;
      for (; key[i] != '\0' && p[i] == key[i]; ++i)
        ;
      if (key[i] == '\0')
        return p + i;
    }
    return nullptr;
  }

  // Tries `name` under each colon separated directory in `list`.
  LIBC_INLINE bool try_path_list(const char *list, const char *name,
                                 LoadedModule &out) {
    if (list == nullptr)
      return false;
    for (const char *segment = list; segment != nullptr;) {
      const char *end = segment;
      while (*end != '\0' && *end != ':')
        ++end;
      char dir[256];
      const size_t length = static_cast<size_t>(end - segment);
      if (length > 0 && length < sizeof(dir)) {
        for (size_t i = 0; i < length; ++i)
          dir[i] = segment[i];
        dir[length] = '\0';
        char path[256];
        if (join(dir, name, path, sizeof(path))) {
          auto loaded = load_module(path, page_size_);
          if (loaded.has_value()) {
            out = loaded.value();
            return true;
          }
        }
      }
      segment = (*end == '\0') ? nullptr : end + 1;
    }
    return false;
  }

  // The search order a loader is expected to use: the requesting object's own
  // run path, then LD_LIBRARY_PATH, then the system directories.
  LIBC_INLINE bool load_dependency(const Module &from, const char *name) {
    if (count_ >= MAX_STARTUP_MODULES) {
      report("too many shared objects\n");
      return false;
    }
    for (const char *p = name; *p != '\0'; ++p) {
      if (*p == '/') {
        auto loaded = load_module(name, page_size_);
        if (!loaded.has_value())
          return missing(name);
        return remember(loaded.value());
      }
    }

    LoadedModule loaded;
    if (try_path_list(run_path(from), name, loaded))
      return remember(loaded);
    if (try_path_list(library_path_, name, loaded))
      return remember(loaded);
    for (const char *dir : DEFAULT_SEARCH_PATHS) {
      char path[256];
      if (!join(dir, name, path, sizeof(path)))
        continue;
      auto found = load_module(path, page_size_);
      if (found.has_value())
        return remember(found.value());
    }
    return missing(name);
  }

  // DT_RUNPATH, or the older DT_RPATH when a module still carries one.
  LIBC_INLINE static const char *run_path(const Module &module) {
    const char *strings = module.strtab();
    if (strings == nullptr)
      return nullptr;
    if (auto offset = module.dynamic().value(DT_RUNPATH))
      return strings + *offset;
    if (auto offset = module.dynamic().value(DT_RPATH))
      return strings + *offset;
    return nullptr;
  }

  LIBC_INLINE bool remember(const LoadedModule &loaded) {
    mappings_[count_] = loaded.mapping;
    modules_[count_] = loaded.module;
    ++count_;
    return true;
  }

  LIBC_INLINE bool missing(const char *name) {
    report("cannot open shared object ");
    report(name);
    report("\n");
    return false;
  }

  LIBC_INLINE static bool join(const char *dir, const char *name, char *out,
                               size_t capacity) {
    size_t n = 0;
    for (const char *p = dir; *p != '\0'; ++p) {
      if (n + 2 >= capacity)
        return false;
      out[n++] = *p;
    }
    out[n++] = '/';
    for (const char *p = name; *p != '\0'; ++p) {
      if (n + 1 >= capacity)
        return false;
      out[n++] = *p;
    }
    out[n] = '\0';
    return true;
  }

  LIBC_INLINE bool setup_tls() {
    auto block = allocate_tls_block(modules_, count_, tls_offsets_,
                                    /*copy_images=*/false);
    if (!block.has_value()) {
      report("cannot allocate thread local storage\n");
      return false;
    }
    tls_block_ = block.value();
    return true;
  }

  LIBC_INLINE bool install_thread_pointer() {
    // Nothing in the process uses thread local storage.
    if (tls_block_.storage == nullptr)
      return true;
    if (!set_thread_pointer(tls_block_.thread_pointer)) {
      report("cannot set the thread pointer\n");
      return false;
    }
    return true;
  }

  // Dependencies are initialised before the objects that need them, so the
  // list is walked backwards. The executable is index zero and is skipped:
  // its own startup code runs its initialisers, and running them here as well
  // would run them twice.
  LIBC_INLINE void run_initialisers() {
    for (size_t i = count_; i > 1; --i)
      run_init_array(modules_[i - 1]);
  }

  LIBC_INLINE static void run_init_array(const Module &module) {
    auto array = module.dynamic().address(DT_INIT_ARRAY);
    auto size = module.dynamic().value(DT_INIT_ARRAYSZ);
    if (!array || !size)
      return;
    auto **functions = reinterpret_cast<void (**)()>(*array);
    for (size_t i = 0; i < *size / sizeof(void *); ++i)
      if (functions[i] != nullptr)
        functions[i]();
  }

  LIBC_INLINE static void report(const char *message) {
    write_to_stderr(message);
  }

  size_t page_size_;
  const char *library_path_;
  Module modules_[MAX_STARTUP_MODULES];
  MappedModule mappings_[MAX_STARTUP_MODULES];
  intptr_t tls_offsets_[MAX_STARTUP_MODULES] = {};
  TlsBlock tls_block_;
  size_t count_ = 0;
};

} // namespace elf
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_ELF_STARTUP_LINKER_H
