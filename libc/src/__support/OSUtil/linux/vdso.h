//===------------- Linux VDSO Header ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_LINUX_VDSO_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_LINUX_VDSO_H
#include "src/__support/common.h"
#include "src/__support/macros/properties/architectures.h"

#if defined(LIBC_TARGET_ARCH_IS_X86)
#include "x86_64/vdso.h"
#elif defined(LIBC_TARGET_ARCH_IS_AARCH64)
#include "aarch64/vdso.h"
#elif defined(LIBC_TARGET_ARCH_IS_ARM)
#include "arm/vdso.h"
#elif defined(LIBC_TARGET_ARCH_IS_RISCV)
#include "riscv/vdso.h"
#endif

namespace LIBC_NAMESPACE {
namespace vdso {
void *get_symbol(VDSOSym);
} // namespace vdso

} // namespace LIBC_NAMESPACE
#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_LINUX_VDSO_H
