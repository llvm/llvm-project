//===-- Classes to capture properites of UEFI applications ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_CONFIG_UEFI_APP_H
#define LLVM_LIBC_CONFIG_UEFI_APP_H

#include "hdr/stdint_proxy.h"
#include "include/llvm-libc-types/EFI_HANDLE.h"
#include "include/llvm-libc-types/EFI_SYSTEM_TABLE.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/architectures.h"

namespace LIBC_NAMESPACE_DECL {

// Data structure which captures properties of a UEFI application.
struct AppProperties {
  // UEFI system table
  EFI_SYSTEM_TABLE *system_table;

  // UEFI image handle
  EFI_HANDLE image_handle;
};

[[gnu::weak]] extern AppProperties app;

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_CONFIG_UEFI_APP_H
