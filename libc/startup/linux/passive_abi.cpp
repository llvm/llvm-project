//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Storage for the loaded module set
///
//===----------------------------------------------------------------------===//

#include "src/__support/elf/passive_abi.h"

#include "src/__support/macros/config.h"

#ifdef LIBC_COPT_SHARED_LIBRARY

namespace LIBC_NAMESPACE_DECL {
namespace elf {
namespace {

Module storage_modules[MAX_PROCESS_MODULES];
MappedModule storage_mappings[MAX_PROCESS_MODULES];
intptr_t storage_tls_offsets[MAX_PROCESS_MODULES];
size_t storage_references[MAX_PROCESS_MODULES];

} // anonymous namespace
} // namespace elf
} // namespace LIBC_NAMESPACE_DECL

// `linked` starts false, so a process that was never linked by this loader
// sees an empty set rather than something that looks usable.
LIBC_NAMESPACE::elf::ModuleSet __llvm_libc_process_modules = {
    LIBC_NAMESPACE::elf::storage_modules,
    LIBC_NAMESPACE::elf::storage_mappings,
    LIBC_NAMESPACE::elf::storage_tls_offsets,
    LIBC_NAMESPACE::elf::storage_references,
    /*count=*/0,
    LIBC_NAMESPACE::elf::MAX_PROCESS_MODULES,
    /*page_size=*/4096,
    /*linked=*/false};

#endif // LIBC_COPT_SHARED_LIBRARY
