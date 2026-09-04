//===-- Implementation of dlclose -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "dlclose.h"

#include "src/__support/CPP/mutex.h"
#include "src/__support/common.h"
#include "src/__support/elf/map_segments.h"
#include "src/__support/elf/passive_abi.h"
#include "src/__support/macros/config.h"
#include "src/dlfcn/dl_internal.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, dlclose, (void *handle)) {
  elf::ModuleSet &set = elf::loaded_modules();
  cpp::lock_guard lock(dl::dl_mutex);

  size_t index = 0;
  if (!set.linked || !dl::index_for(handle, index)) {
    dl::set_error("dlclose was given a handle that dlopen did not return");
    return -1;
  }
  if (set.references[index] == 0) {
    dl::set_error("shared object is already closed");
    return -1;
  }

  if (--set.references[index] != 0)
    return 0;

  // The objects the process started with are never unloaded, so their count
  // is left where it is rather than allowed to reach zero.
  if (index == 0) {
    set.references[index] = 1;
    return 0;
  }

  // Only the last module can be dropped from the set, since a handle is an
  // index into it and removing from the middle would renumber the rest. An
  // earlier one keeps its mapping until the process ends, which is allowed:
  // dlclose is not required to unload anything. Nor is it finalized, since
  // opening it again hands back the same module without initializing it.
  if (index + 1 == set.count) {
    elf::run_fini_array(set.modules[index]);
    elf::unmap_module(set.mappings[index]);
    set.count = index;
  }
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
