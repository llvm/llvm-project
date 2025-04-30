/* Copyright (C) 2001-2021 Free Software Foundation, Inc.
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

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ucontext.h>
#include <unistd.h>

static ucontext_t ctx[3];

static int was_in_f1;
static int was_in_f2;

static char st2[32768];

static volatile int flag;

static void
f1 (int a00, int a01, int a02, int a03, int a04, int a05, int a06, int a07,
    int a08, int a09, int a10, int a11, int a12, int a13, int a14, int a15,
    int a16, int a17, int a18, int a19, int a20, int a21, int a22, int a23,
    int a24, int a25, int a26, int a27, int a28, int a29, int a30, int a31,
    int a32)
{
  printf ("start f1(a00=%08x,a01=%08x,a02=%08x,a03=%08x,\n"
	  "         a04=%08x,a05=%08x,a06=%08x,a07=%08x,\n"
	  "         a08=%08x,a09=%08x,a10=%08x,a11=%08x,\n"
	  "         a12=%08x,a13=%08x,a14=%08x,a15=%08x,\n"
	  "         a16=%08x,a17=%08x,a18=%08x,a19=%08x,\n"
	  "         a20=%08x,a21=%08x,a22=%08x,a23=%08x,\n"
	  "         a24=%08x,a25=%08x,a26=%08x,a27=%08x,\n"
	  "         a28=%08x,a29=%08x,a30=%08x,a31=%08x,\n"
	  "         a32=%08x) [%d]\n",
	  a00, a01, a02, a03, a04, a05, a06, a07,
	  a08, a09, a10, a11, a12, a13, a14, a15,
	  a16, a17, a18, a19, a20, a21, a22, a23,
	  a24, a25, a26, a27, a28, a29, a30, a31,
	  a32, flag);

  if (a00 != (0x00000001 << flag) || a01 != (0x00000004 << flag)
      || a02 != (0x00000012 << flag) || a03 != (0x00000048 << flag)
      || a04 != (0x00000123 << flag) || a05 != (0x0000048d << flag)
      || a06 != (0x00001234 << flag) || a07 != (0x000048d1 << flag)
      || a08 != (0x00012345 << flag) || a09 != (0x00048d15 << flag)
      || a10 != (0x00123456 << flag) || a11 != (0x0048d159 << flag)
      || a12 != (0x01234567 << flag) || a13 != (0x048d159e << flag)
      || a14 != (0x12345678 << flag) || a15 != (0x48d159e2 << flag)
      || a16 != (0x23456789 << flag) || a17 != (0x8d159e26 << flag)
      || a18 != (0x3456789a << flag) || a19 != (0xd159e26a << flag)
      || a20 != (0x456789ab << flag) || a21 != (0x159e26af << flag)
      || a22 != (0x56789abc << flag) || a23 != (0x59e26af3 << flag)
      || a24 != (0x6789abcd << flag) || a25 != (0x9e26af37 << flag)
      || a26 != (0x789abcde << flag) || a27 != (0xe26af37b << flag)
      || a28 != (0x89abcdef << flag) || a29 != (0x26af37bc << flag)
      || a30 != (0x9abcdef0 << flag) || a31 != (0x6af37bc3 << flag)
      || a32 != (0xabcdef0f << flag))
    {
      puts ("arg mismatch");
      exit (-1);
    }

  if (flag && swapcontext (&ctx[1], &ctx[2]) != 0)
    {
      printf ("%s: swapcontext: %m\n", __FUNCTION__);
      exit (1);
    }
  printf ("finish f1 [%d]\n", flag);
  flag++;
  was_in_f1++;
}

static void
f2 (void)
{
  puts ("start f2");
  if (swapcontext (&ctx[2], &ctx[1]) != 0)
    {
      printf ("%s: swapcontext: %m\n", __FUNCTION__);
      exit (1);
    }
  puts ("finish f2");
  was_in_f2 = 1;
}

volatile int global;


static int back_in_main;


static void
check_called (void)
{
  if (back_in_main == 0)
    {
      puts ("program did not reach main again");
      _exit (1);
    }
}


int
main (void)
{
  atexit (check_called);

  char st1[32768];

  puts ("making contexts");
  if (getcontext (&ctx[0]) != 0)
    {
      if (errno == ENOSYS)
	{
	  back_in_main = 1;
	  exit (0);
	}

      printf ("%s: getcontext: %m\n", __FUNCTION__);
      exit (1);
    }

  if (getcontext (&ctx[1]) != 0)
    {
      printf ("%s: getcontext: %m\n", __FUNCTION__);
      exit (1);
    }

  ctx[1].uc_stack.ss_sp = st1;
  ctx[1].uc_stack.ss_size = sizeof st1;
  ctx[1].uc_link = &ctx[0];
  errno = 0;
  makecontext (&ctx[1], (void (*) (void)) f1, 33,
	       0x00000001 << flag, 0x00000004 << flag,
	       0x00000012 << flag, 0x00000048 << flag,
	       0x00000123 << flag, 0x0000048d << flag,
	       0x00001234 << flag, 0x000048d1 << flag,
	       0x00012345 << flag, 0x00048d15 << flag,
	       0x00123456 << flag, 0x0048d159 << flag,
	       0x01234567 << flag, 0x048d159e << flag,
	       0x12345678 << flag, 0x48d159e2 << flag,
	       0x23456789 << flag, 0x8d159e26 << flag,
	       0x3456789a << flag, 0xd159e26a << flag,
	       0x456789ab << flag, 0x159e26af << flag,
	       0x56789abc << flag, 0x59e26af3 << flag,
	       0x6789abcd << flag, 0x9e26af37 << flag,
	       0x789abcde << flag, 0xe26af37b << flag,
	       0x89abcdef << flag, 0x26af37bc << flag,
	       0x9abcdef0 << flag, 0x6af37bc3 << flag,
	       0xabcdef0f << flag);

  /* Without this check, a stub makecontext can make us spin forever.  */
  if (errno == ENOSYS)
    {
      puts ("makecontext not implemented");
      back_in_main = 1;
      return 0;
    }

  /* Play some tricks with this context.  */
  if (++global == 1)
    if (setcontext (&ctx[1]) != 0)
      {
	printf ("%s: setcontext: %m\n", __FUNCTION__);
	exit (1);
      }
  if (global != 2)
    {
      printf ("%s: 'global' not incremented twice\n", __FUNCTION__);
      exit (1);
    }

  if (getcontext (&ctx[2]) != 0)
    {
      printf ("%s: second getcontext: %m\n", __FUNCTION__);
      exit (1);
    }
  ctx[2].uc_stack.ss_sp = st2;
  ctx[2].uc_stack.ss_size = sizeof st2;
  ctx[2].uc_link = &ctx[1];
  makecontext (&ctx[2], f2, 0);

  puts ("swapping contexts");
  if (swapcontext (&ctx[0], &ctx[2]) != 0)
    {
      printf ("%s: swapcontext: %m\n", __FUNCTION__);
      exit (1);
    }
  puts ("back at main program");
  back_in_main = 1;

  if (was_in_f1 < 2)
    {
      puts ("didn't reach f1 twice");
      exit (1);
    }
  if (was_in_f2 == 0)
    {
      puts ("didn't reach f2");
      exit (1);
    }

  puts ("test succeeded");
  return 0;
}
