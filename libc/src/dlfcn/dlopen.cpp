//===-- Implementation of dlopen ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "dlopen.h"

#include "src/__support/CPP/mutex.h"
#include "src/__support/common.h"
#include "src/__support/elf/bind.h"
#include "src/__support/elf/load_module.h"
#include "src/__support/elf/passive_abi.h"
#include "src/__support/macros/config.h"
#include "src/dlfcn/dl_internal.h"

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
        same_name(set.modules[i].soname(), path)) {
      ++set.references[i];
      return dl::handle_for(i);
    }
  }

  if (set.count >= set.capacity) {
    dl::set_error("too many shared objects are open");
    return nullptr;
  }

  auto loaded = elf::load_module(path, set.page_size);
  if (!loaded.has_value()) {
    dl::set_error(path, "cannot open shared object file");
    return nullptr;
  }

  // Thread local storage in a newly opened object would have to be added to
  // every thread that already exists, which needs a dynamic thread vector.
  // Refusing is better than handing back a module whose thread locals alias
  // something else.
  if (loaded.value().module.tls() != nullptr) {
    elf::unmap_module(loaded.value().mapping);
    dl::set_error("shared object uses thread local storage");
    return nullptr;
  }

  const size_t index = set.count;
  set.modules[index] = loaded.value().module;
  set.mappings[index] = loaded.value().mapping;
  set.tls_offsets[index] = 0;
  set.references[index] = 1;
  set.count = index + 1;

  // Bind against everything now loaded, the new object included, so it can
  // refer to itself as well as to what was already there.
  elf::SearchOrder order(set.modules, set.count);
  order.set_tls_offsets(set.tls_offsets);
  elf::BindResult result = elf::bind_module(set.modules[index], index, order);
  if (result.unresolved != 0) {
    set.count = index;
    elf::unmap_module(loaded.value().mapping);
    dl::set_error("shared object has undefined symbols");
    return nullptr;
  }

  run_init_array(set.modules[index]);
  return dl::handle_for(index);
}

} // namespace LIBC_NAMESPACE_DECL
