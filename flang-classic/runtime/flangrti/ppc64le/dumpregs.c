/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "flangrti_config.h"
#if defined(HAVE_GREGSET_T)
#include <sys/ucontext.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <ctype.h>
#include "stdioInterf.h"

typedef struct {
    int     rn;     // Register index in to "regs" pointer
    char    *s;     // Symbolic name of register
} gprs_t;


/*
 * The way the structure below is organized, the registers are all
 * sequential with no gaps - the structure is probably overkill - but
 * allows for some flexibility.
 */

gprs_t gprs[] = {
//    { 0, "r0" }, { 1, "r1" }, { 2, "r2" }, { 3, "r3" }, { 4, "r4" },
    { 0, "r0" }, { 1, "sp" }, { 2, "toc"}, { 3, "r3" }, { 4, "r4" },
    { 5, "r5" }, { 6, "r6" }, { 7, "r7" }, { 8, "r8" }, { 9, "r9" },
    {10, "r10"}, {11, "r11"}, {12, "r12"}, {13, "r13"}, {14, "r14"},
    {15, "r15"}, {16, "r16"}, {17, "r17"}, {18, "r18"}, {19, "r19"},
    {20, "r20"}, {21, "r21"}, {22, "r22"}, {23, "r23"}, {24, "r24"},
    {25, "r25"}, {26, "r26"}, {27, "r27"}, {28, "r28"}, {29, "r29"},
    {30, "r30"}, {31, "r31"},
    {PT_NIP, "pc"},
    {PT_MSR, "msr"},
    {PT_ORIG_R3, "orig_r3"},
    {PT_CTR, "ctr"},
    {PT_LNK, "lr"},
    {PT_XER, "xer"},
    {PT_CCR, "cr"},
    {PT_SOFTE, "softe"},
    {PT_TRAP, "trap"},
    {PT_DAR, "dar"},
    {PT_DSISR, "dsisr"},
    {PT_RESULT, "result"},
};

void
dumpregs(uint64_t *regs)
{
  int i;
  int j;
  char *pc;

  if (regs == NULL)
    return;             // Not sure if this is possible

/*
 * Output has the following format:
 *  <REG>    <HEXADECIMAL>         <DECIMAL>               <ASCII>
 *  Example:
 *  r0       0x00003fffaf4a309c       70367390085276       .0J..?..
 *  sp       0x00003ffff437d1a0       70368546509216       ..7..?..
 *  toc      0x0000000010019300            268538624       ........
 *  r3       0x0000000010000e64            268439140       d.......
 *  ...
 */

  for (i = 0 ; i < sizeof gprs / sizeof *gprs ; ++i) {
    fprintf(__io_stderr(), " %-8s 0x%016" PRIx64 " %20" PRId64 "\t",
      gprs[i].s, regs[gprs[i].rn], regs[gprs[i].rn]);
    pc = (char *)&(regs[gprs[i].rn]);
    for(j = 0 ; j < 8 ; ++j) {
      fputc(isprint(pc[j]) ? pc[j] : '.', __io_stderr());
    }
    fputs("\n", __io_stderr());
  }

}

uint64_t *
getRegs(ucontext_t *u)
{
  mcontext_t *mc = &u->uc_mcontext;
  return (uint64_t *)&(mc->gp_regs);
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
