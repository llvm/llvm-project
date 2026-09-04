//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// The widest ELF integer types for the class being built
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_ELF_ELF_TYPES_H
#define LLVM_LIBC_SRC___SUPPORT_ELF_ELF_TYPES_H

#include "hdr/elf_proxy.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace elf {

// ELF32 has no Xword or Sxword, so ElfW(Xword) names a type that does not
// exist there. Its 32 bit words are the widest it has, and are what the
// fields these stand for are made of.
#ifdef __LP64__
using Xword = Elf64_Xword;
using Sxword = Elf64_Sxword;
#else
using Xword = Elf32_Word;
using Sxword = Elf32_Sword;
#endif

} // namespace elf
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_ELF_ELF_TYPES_H
