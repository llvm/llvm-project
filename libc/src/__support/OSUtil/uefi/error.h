//===---------- UEFI implementation of error utils ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_UEFI_ERROR_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_UEFI_ERROR_H

#include "include/llvm-libc-types/EFI_STATUS.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

int uefi_status_to_errno(EFI_STATUS status);
EFI_STATUS errno_to_uefi_status(int errno);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_UEFI_ERROR_H
