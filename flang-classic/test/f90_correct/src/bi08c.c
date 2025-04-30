/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

extern void mm2(void);
extern void mM4(void);
extern void ss2(void);
extern void sS4(void);

void
cc(void) {
  mm2();
  mM4();
  ss2();
  sS4();
}
