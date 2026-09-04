//===-- Implementation of dlsym -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "dlsym.h"

#include "src/__support/CPP/mutex.h"
#include "src/__support/common.h"
#include "src/__support/elf/bind.h"
#include "src/__support/elf/passive_abi.h"
#include "src/__support/macros/config.h"
#include "src/dlfcn/dl_internal.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(void *, dlsym,
                   (void *__restrict handle, const char *__restrict name)) {
  elf::ModuleSet &set = elf::loaded_modules();
  cpp::lock_guard lock(dl::dl_mutex);

  if (!set.linked) {
    dl::set_error("dlsym is only available in a dynamically linked process");
    return nullptr;
  }
  if (name == nullptr) {
    dl::set_error("dlsym was given no symbol name");
    return nullptr;
  }

  size_t index = 0;
  if (!dl::index_for(handle, index)) {
    dl::set_error("dlsym was given a handle that dlopen did not return");
    return nullptr;
  }

  const elf::Module &module = set.modules[index];
  const ElfW(Sym) *symbol = module.symbols().lookup(name);
  // A symbol the object only references does not count as finding it.
  if (symbol == nullptr || symbol->st_shndx == SHN_UNDEF) {
    dl::set_error("symbol not found");
    return nullptr;
  }
  return reinterpret_cast<void *>(module.load_bias() + symbol->st_value);
}

} // namespace LIBC_NAMESPACE_DECL
