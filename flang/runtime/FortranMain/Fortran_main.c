//===-- runtime/FortranMain/Fortran_main.c --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/main.h"
#include "flang/Runtime/stop.h"

/* main entry into PROGRAM */
void _QQmain();

/* C main stub */
int main(int argc, const char *argv[], const char *envp[]) {
  RTNAME(ProgramStart)(argc, argv, envp);
  _QQmain();
  RTNAME(ProgramEndStatement)();
  return 0;
}
