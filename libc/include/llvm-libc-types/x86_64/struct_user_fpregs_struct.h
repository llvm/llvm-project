//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definition of struct user_fpregs_struct for x86_64.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TYPES_X86_64_STRUCT_USER_FPREGS_STRUCT_H
#define LLVM_LIBC_TYPES_X86_64_STRUCT_USER_FPREGS_STRUCT_H

// The 512-byte FXSAVE area, which the kernel exposes through PTRACE_GETFPREGS
// and the NT_PRFPREG core dump note.
struct user_fpregs_struct {
  unsigned short cwd;
  unsigned short swd;
  unsigned short ftw;
  unsigned short fop;
  unsigned long long rip;
  unsigned long long rdp;
  unsigned int mxcsr;
  unsigned int mxcr_mask;
  unsigned int st_space[32];
  unsigned int xmm_space[64];
  unsigned int padding[24];
};

#endif // LLVM_LIBC_TYPES_X86_64_STRUCT_USER_FPREGS_STRUCT_H
