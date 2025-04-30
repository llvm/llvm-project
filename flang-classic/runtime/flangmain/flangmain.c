/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "FuncArgMacros.h"

#if defined(__PGI)
#pragma global - Mx, 119, 2048
#pragma global - x 119 0x10000000
#pragma global - x 129 0x200
#endif

extern char **__io_environ();
extern void __io_set_argc(int);
extern void __io_set_argv(char **);

int main(int argc, char** argv)
{
  int i = 0;

  __io_set_argc(argc);
  __io_set_argv(argv);

  MAIN_(argc, argv, __io_environ());
  ENTF90(EXIT, exit)(&i);
}
