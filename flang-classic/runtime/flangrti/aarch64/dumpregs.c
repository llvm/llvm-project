/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "dumpregs.h"
#if defined(HAVE_GREGSET_T)
#include <sys/ucontext.h>
#include <stddef.h>
#include <stdioInterf.h>
#include <stdint.h>
#include <inttypes.h>
#include <ctype.h>


typedef struct {
    size_t  ro;     // Register offset in to mcontext_t structure
    const char *s;  // Symbolic name of register
} xregs_t;


/*
 * The way the structure below is organized, the X registers are all
 * sequential with no gaps - the structure is probably overkill - but
 * allows for some flexibility.
 */

xregs_t xregs[] = {
    { offsetof(mcontext_t, regs[0])/sizeof(uint64_t), "x0" },
    { offsetof(mcontext_t, regs[1])/sizeof(uint64_t), "x1" },
    { offsetof(mcontext_t, regs[2])/sizeof(uint64_t), "x2" },
    { offsetof(mcontext_t, regs[3])/sizeof(uint64_t), "x3" },
    { offsetof(mcontext_t, regs[4])/sizeof(uint64_t), "x4" },
    { offsetof(mcontext_t, regs[5])/sizeof(uint64_t), "x5" },
    { offsetof(mcontext_t, regs[6])/sizeof(uint64_t), "x6" },
    { offsetof(mcontext_t, regs[7])/sizeof(uint64_t), "x7" },
    { offsetof(mcontext_t, regs[8])/sizeof(uint64_t), "x8" },
    { offsetof(mcontext_t, regs[9])/sizeof(uint64_t), "x9" },
    { offsetof(mcontext_t, regs[10])/sizeof(uint64_t), "x10" },
    { offsetof(mcontext_t, regs[11])/sizeof(uint64_t), "x11" },
    { offsetof(mcontext_t, regs[12])/sizeof(uint64_t), "x12" },
    { offsetof(mcontext_t, regs[13])/sizeof(uint64_t), "x13" },
    { offsetof(mcontext_t, regs[14])/sizeof(uint64_t), "x14" },
    { offsetof(mcontext_t, regs[15])/sizeof(uint64_t), "x15" },
    { offsetof(mcontext_t, regs[16])/sizeof(uint64_t), "x16" },
    { offsetof(mcontext_t, regs[17])/sizeof(uint64_t), "x17" },
    { offsetof(mcontext_t, regs[18])/sizeof(uint64_t), "x18" },
    { offsetof(mcontext_t, regs[19])/sizeof(uint64_t), "x19" },
    { offsetof(mcontext_t, regs[20])/sizeof(uint64_t), "x20" },
    { offsetof(mcontext_t, regs[21])/sizeof(uint64_t), "x21" },
    { offsetof(mcontext_t, regs[22])/sizeof(uint64_t), "x22" },
    { offsetof(mcontext_t, regs[23])/sizeof(uint64_t), "x23" },
    { offsetof(mcontext_t, regs[24])/sizeof(uint64_t), "x24" },
    { offsetof(mcontext_t, regs[25])/sizeof(uint64_t), "x25" },
    { offsetof(mcontext_t, regs[26])/sizeof(uint64_t), "x26" },
    { offsetof(mcontext_t, regs[27])/sizeof(uint64_t), "x27" },
    { offsetof(mcontext_t, regs[28])/sizeof(uint64_t), "x28" },
    { offsetof(mcontext_t, regs[29])/sizeof(uint64_t), "x29" },
    { offsetof(mcontext_t, regs[30])/sizeof(uint64_t), "x30" },
    { offsetof(mcontext_t, sp)/sizeof(uint64_t), "sp"},
    { offsetof(mcontext_t, pc)/sizeof(uint64_t), "pc"},
    { offsetof(mcontext_t, pstate)/sizeof(uint64_t), "pstate"},
    { offsetof(mcontext_t, fault_address)/sizeof(uint64_t), "fault_addr"},
};

void
dumpregs(gregset_t *mc)
{
  int i;
  int j;
  char *pc;
  uint64_t *regs;

  if (mc == NULL)
    return;             // Not sure if this is possible

  regs = mc;

/*
 * Output has the following format:
 *  <REG>    <HEXADECIMAL>         <DECIMAL>               <ASCII>
 *  Example:
 *  x0       0x0000000000000001                    1        ........
 *  x1       0x0000000000000000                    0        ........
 *  x2       0x000003ffffffe9c0        4398046505408        ........
 *  x3       0x000003ffb7bc8254        4396834128468        T.......
 * ...
 *  x29      0x000003ffffffe750        4398046504784        P.......
 *  x30      0x000003ffb7bc84ac        4396834129068        ........
 *  sp       0x000003ffffffe750        4398046504784        P.......
 *  pc       0x000003ffb7bfd0e0        4396834345184        ........
 *  pstate   0x0000000080000000           2147483648        ........
 *  fault_addr 0x0000000000000000                    0      ........
 */

  for (i = 0 ; (size_t)i < sizeof(xregs) / sizeof(*xregs); ++i) {
    fprintf(__io_stderr(), "%-8s 0x%016" PRIx64 " %20" PRId64 "\t",
      xregs[i].s, regs[xregs[i].ro], regs[xregs[i].ro]);
    pc = (char *)&(regs[xregs[i].ro]);
    for(j = 0 ; j < 8 ; ++j) {
      fputc(isprint(pc[j]) ? pc[j] : '.', __io_stderr());
    }
    fputs("\n", __io_stderr());
  }

}

gregset_t *
getRegs(ucontext_t *u)
{
  mcontext_t *mc = &u->uc_mcontext;
  return (gregset_t *)mc;
}

#else

void
dumpregs(void *regs)
{
}

void *
getRegs(void *u)
{
  return (void *)0;
}

#endif
