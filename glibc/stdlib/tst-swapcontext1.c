/* Check multiple makecontext calls.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <stdio.h>
#include <stdlib.h>
#include <ucontext.h>

static ucontext_t uctx_main, uctx_func1, uctx_func2;
const char *str1 = "\e[31mswapcontext(&uctx_func1, &uctx_main)\e[0m";
const char *str2 = "\e[34mswapcontext(&uctx_func2, &uctx_main)\e[0m";
const char *fmt1 = "\e[31m";
const char *fmt2 = "\e[34m";

#define handle_error(msg) \
  do { perror(msg); exit(EXIT_FAILURE); } while (0)

__attribute__((noinline, noclone))
static void
func4(ucontext_t *uocp, ucontext_t *ucp, const char *str, const char *fmt)
{
  printf("      %sfunc4: %s\e[0m\n", fmt, str);
  if (swapcontext(uocp, ucp) == -1)
    handle_error("swapcontext");
  printf("      %sfunc4: returning\e[0m\n", fmt);
}

__attribute__((noinline, noclone))
static void
func3(ucontext_t *uocp, ucontext_t *ucp, const char *str, const char *fmt)
{
  printf("    %sfunc3: func4(uocp, ucp, str)\e[0m\n", fmt);
  func4(uocp, ucp, str, fmt);
  printf("    %sfunc3: returning\e[0m\n", fmt);
}

__attribute__((noinline, noclone))
static void
func1(void)
{
  while ( 1 )
    {
      printf("  \e[31mfunc1: func3(&uctx_func1, &uctx_main, str1)\e[0m\n");
      func3( &uctx_func1, &uctx_main, str1, fmt1);
    }
}

__attribute__((noinline, noclone))
static void
func2(void)
{
  while ( 1 )
    {
      printf("  \e[34mfunc2: func3(&uctx_func2, &uctx_main, str2)\e[0m\n");
      func3(&uctx_func2, &uctx_main, str2, fmt2);
    }
}

static int
do_test (void)
{
  char func1_stack[16384];
  char func2_stack[16384];
  int i;

  if (getcontext(&uctx_func1) == -1)
    handle_error("getcontext");
  uctx_func1.uc_stack.ss_sp = func1_stack;
  uctx_func1.uc_stack.ss_size = sizeof (func1_stack);
  uctx_func1.uc_link = &uctx_main;
  makecontext(&uctx_func1, func1, 0);

  if (getcontext(&uctx_func2) == -1)
    handle_error("getcontext");
  uctx_func2.uc_stack.ss_sp = func2_stack;
  uctx_func2.uc_stack.ss_size = sizeof (func2_stack);
  uctx_func2.uc_link = &uctx_func1;
  makecontext(&uctx_func2, func2, 0);

  for ( i = 0; i < 4; i++ )
    {
      if (swapcontext(&uctx_main, &uctx_func1) == -1)
	handle_error("swapcontext");
      printf("        \e[35mmain: swapcontext(&uctx_main, &uctx_func2)\n\e[0m");
      if (swapcontext(&uctx_main, &uctx_func2) == -1)
	handle_error("swapcontext");
      printf("        \e[35mmain: swapcontext(&uctx_main, &uctx_func1)\n\e[0m");
    }

  printf("main: exiting\n");
  exit(EXIT_SUCCESS);
}

#include <support/test-driver.c>
