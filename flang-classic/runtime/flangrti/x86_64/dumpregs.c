/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "dumpregs.h"
#include <stdint.h>
#include <inttypes.h>
#include "stdioInterf.h"

/* define register indexes */

#define R8 0
#define R9 1
#define R10 2
#define R11 3
#define R12 4
#define R13 5
#define R14 6
#define R15 7
#define RDI 8
#define RSI 9
#define RBP 10
#define RBX 11
#define RDX 12
#define RAX 13
#define RCX 14
#define RSP 15
#define RIP 16

#if !defined(HAVE_GREGSET_T)
/* no gregs and/or ucontext defined in for OSX or Windows */
void * 
getRegs(void *u)
{
 return (void *)0;
}
void
dumpregs(void *regs)
{
}

#else

#if defined(TARGET_LINUX_X8664)
/*
 * greg2u64: convert value of typedef greg_t to uint64_t.
 * Avoids recasting through pointers.
 */
static inline uint64_t greg2u64(greg_t r)
{
  struct {
    union {
      greg_t g;
      uint64_t u64;
    };
  } g2u;
  g2u.g = r;
  return g2u.u64;
}
#endif

void
dumpregs(gregset_t *regs)
{
#if defined(TARGET_LINUX_X8664)

/*
 * Using "0x%016" PRIx64 instead of "%#016" PRIx64 to keep output
 * aligned between rows and columns.  Probably unnecessary, but provides
 * for a consistent appearance.
 */

  fprintf(__io_stderr(), "   rax 0x%016" PRIx64 ", rbx 0x%016" PRIx64 ", rcx 0x%016" PRIx64 "\n",
          greg2u64((*regs)[RAX]), greg2u64((*regs)[RBX]), greg2u64((*regs)[RCX]));
  fprintf(__io_stderr(), "   rdx 0x%016" PRIx64 ", rsp 0x%016" PRIx64 ", rbp 0x%016" PRIx64 "\n",
          greg2u64((*regs)[RDX]), greg2u64((*regs)[RSP]), greg2u64((*regs)[RBP]));
  fprintf(__io_stderr(), "   rsi 0x%016" PRIx64 ", rdi 0x%016" PRIx64 ", r8  0x%016" PRIx64 "\n",
          greg2u64((*regs)[RSI]), greg2u64((*regs)[RDI]), greg2u64((*regs)[R8]));
  fprintf(__io_stderr(), "   r9  0x%016" PRIx64 ", r10 0x%016" PRIx64 ", r11 0x%016" PRIx64 "\n",
          greg2u64((*regs)[R9]), greg2u64((*regs)[R10]), greg2u64((*regs)[R11]));
  fprintf(__io_stderr(), "   r12 0x%016" PRIx64 ", r13 0x%016" PRIx64 ", r14 0x%016" PRIx64 "\n",
          greg2u64((*regs)[R12]), greg2u64((*regs)[R13]), greg2u64((*regs)[R14]));
  fprintf(__io_stderr(), "   r15 0x%016" PRIx64 "\n", greg2u64((*regs)[R15]));
#endif
}

gregset_t *
getRegs(ucontext_t *u)
{
  return &u->uc_mcontext.gregs;
}

#endif

