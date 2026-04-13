//===-- Definition of type mcontext_t -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Note: Definitions in this file are based on the Linux kernel ABI.

#ifndef LLVM_LIBC_TYPES_MCONTEXT_T_H
#define LLVM_LIBC_TYPES_MCONTEXT_T_H

// The following definitions correspond to the general purpose registers.
// The layout of gregset_t and the enum indices must match the layout of
// 'struct sigcontext' in the Linux kernel on x86_64 (see
// arch/x86/include/uapi/asm/sigcontext.h). The kernel uses named fields
// (like r8, r9) while we use an array indexed by these enum values.
//
// Note: The kernel defines segment registers (cs, gs, fs, ss) as four
// separate 16-bit fields. In our flat 64-bit array representation, they
// are packed into a single 64-bit slot at index __LIBC_REG_CSGSFS, occupying
// the exact same 8 bytes of memory.
typedef long long int greg_t;
typedef greg_t gregset_t[23];

enum {
  __LIBC_REG_R8 = 0,
  __LIBC_REG_R9,
  __LIBC_REG_R10,
  __LIBC_REG_R11,
  __LIBC_REG_R12,
  __LIBC_REG_R13,
  __LIBC_REG_R14,
  __LIBC_REG_R15,
  __LIBC_REG_RDI,
  __LIBC_REG_RSI,
  __LIBC_REG_RBP,
  __LIBC_REG_RBX,
  __LIBC_REG_RDX,
  __LIBC_REG_RAX,
  __LIBC_REG_RCX,
  __LIBC_REG_RSP,
  __LIBC_REG_RIP,
  __LIBC_REG_EFL,
  __LIBC_REG_CSGSFS,
  __LIBC_REG_ERR,
  __LIBC_REG_TRAPNO,
  __LIBC_REG_OLDMASK,
  __LIBC_REG_CR2
};

// The following structures (_libc_fpxreg, _libc_xmmreg, _libc_fpstate)
// represent the floating-point state and must match the layout used by the
// x86 FXSAVE instruction and the kernel's 'struct _fpstate' (see
// arch/x86/include/uapi/asm/sigcontext.h).
struct _libc_fpxreg {
  unsigned short significand[4];
  unsigned short exponent;
  unsigned short padding[3];
};

struct _libc_xmmreg {
  unsigned int element[4];
};

// Note: The kernel's 'struct _fpstate' uses flat arrays like 'st_space[32]'
// and 'xmm_space[64]'. We use structured arrays '_st[8]' and '_xmm[16]'
// instead to allow focused access, but the memory layout and total sizes
// (128 bytes for _st and 256 bytes for _xmm) are identical to FXSAVE.
struct _libc_fpstate {
  unsigned short cwd;
  unsigned short swd;
  unsigned short ftw; // Maps to 'twd' (Tag Word) in the kernel's _fpstate.
  unsigned short fop;
  unsigned long long rip;
  unsigned long long rdp;
  unsigned int mxcsr;
  unsigned int mxcr_mask;
  struct _libc_fpxreg _st[8];
  struct _libc_xmmreg _xmm[16];
  unsigned int padding[24];
};

// fpregset_t is the type used to represent the floating-point register set.
// On x86_64, this is defined as a pointer to the state structure, matching
// the 'fpstate' pointer in the kernel's sigcontext.
typedef struct _libc_fpstate *fpregset_t;

// mcontext_t represents the machine state. This structure must match the
// layout of 'struct sigcontext' in the Linux kernel on x86_64.
typedef struct {
  gregset_t gregs;
  fpregset_t fpregs;
  unsigned long long __reserved1[8];
} mcontext_t;

#endif // LLVM_LIBC_TYPES_MCONTEXT_T_H
