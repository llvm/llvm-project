/* Test the protected visibility when main is linked with modb and moda
   in that order:
   1. Protected symbols, protected1, protected2 and protected3, defined
      in moda, are used in moda.
   2.  Protected symbol, protected3, defined in modb, are used in modb
   3. Symbol, protected1, defined in modb, is used in main and modb.
   4. Symbol, protected2, defined in main, is used in main.
   5. Symbol, protected3, defined in modb, is also used in main.

   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

/* This file must be compiled as PIE to avoid copy relocation when
   accessing protected symbols defined in shared libaries since copy
   relocation doesn't work with protected symbols and linker in
   binutils 2.26 enforces this rule.  */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tst-protected1mod.h"

/* Prototype for our test function.  */
extern int do_test (void);

int protected2 = -1;

/* This defines the `main' function and some more.  */
#include <support/test-driver.c>

int
do_test (void)
{
  int res = 0;

  /* Check if we get the same address for the protected data symbol.  */
  if (&protected1 == protected1a_p ())
    {
      puts ("`protected1' in main and moda has  same address");
      res = 1;
    }
  if (&protected1 != protected1b_p ())
    {
      puts ("`protected1' in main and modb doesn't have same address");
      res = 1;
    }

  /* Check if we get the right value for the protected data symbol.  */
  if (protected1 != -3)
    {
      puts ("`protected1' in main and modb doesn't have same value");
      res = 1;
    }

  /* Check if we get the right value for data defined in executable.  */
  if (protected2 != -1)
    {
      puts ("`protected2' in main has the wrong value");
      res = 1;
    }

  /* Check `protected1' in moda.  */
  if (!check_protected1 ())
    {
      puts ("`protected1' in moda has the wrong value");
      res = 1;
    }

  /* Check `protected2' in moda.  */
  if (!check_protected2 ())
    {
      puts ("`protected2' in moda has the wrong value");
      res = 1;
    }

  /* Check if we get the same address for the protected data symbol.  */
  if (&protected3 == protected3a_p ())
    {
      puts ("`protected3' in main and moda has same address");
      res = 1;
    }
  if (&protected3 != protected3b_p ())
    {
      puts ("`protected3' in main and modb doesn't have same address");
      res = 1;
    }

  /* Check if we get the right value for the protected data symbol.  */
  if (protected3 != -5)
    {
      puts ("`protected3' in main and modb doesn't have same value");
      res = 1;
    }

  /* Check `protected3' in moda.  */
  if (!check_protected3a ())
    {
      puts ("`protected3' in moda has the wrong value");
      res = 1;
    }

  /* Check `protected3' in modb.  */
  if (!check_protected3b ())
    {
      puts ("`protected3' in modb has the wrong value");
      res = 1;
    }

  /* Set `protected2' in moda to 30.  */
  set_protected2 (300);

  /* Check `protected2' in moda.  */
  if (!check_protected2 ())
    {
      puts ("`protected2' in moda has the wrong value");
      res = 1;
    }

  /* Check if we get the right value for data defined in executable.  */
  if (protected2 != -1)
    {
      puts ("`protected2' in main has the wrong value");
      res = 1;
    }

  /* Set `protected1' in moda to 30.  */
  set_protected1a (30);

  /* Check `protected1' in moda.  */
  if (!check_protected1 ())
    {
      puts ("`protected1' in moda has the wrong value");
      res = 1;
    }

  /* Check if we get the same value for the protected data symbol.  */
  if (protected1 != -3)
    {
      puts ("`protected1' in main has the wrong value");
      res = 1;
    }

  protected2 = -300;

  /* Check `protected2' in moda.  */
  if (!check_protected2 ())
    {
      puts ("`protected2' in moda has the wrong value");
      res = 1;
    }

  /* Check if data defined in executable is changed.  */
  if (protected2 != -300)
    {
      puts ("`protected2' in main is changed");
      res = 1;
    }

  /* Set `protected1' in modb to 40.  */
  set_protected1b (40);

  /* Check `protected1' in moda.  */
  if (!check_protected1 ())
    {
      puts ("`protected1' in moda has the wrong value");
      res = 1;
    }

  /* Check if we get the updated value for the protected data symbol.  */
  if (protected1 != 40)
    {
      puts ("`protected1' in main doesn't have the updated value");
      res = 1;
    }

  /* Set `protected3' in moda to 80.  */
  set_protected3a (80);

  /* Check `protected3' in moda.  */
  if (!check_protected3a ())
    {
      puts ("`protected3' in moda has the wrong value");
      res = 1;
    }

  /* Check if we get the updated value for the protected data symbol.  */
  if (protected3 != -5)
    {
      puts ("`protected3' in main doesn't have the updated value");
      res = 1;
    }

  /* Check `protected3' in modb.  */
  if (!check_protected3b ())
    {
      puts ("`protected3' in modb has the wrong value");
      res = 1;
    }

  /* Set `protected3' in modb to 100.  */
  set_protected3b (100);

  /* Check `protected3' in moda.  */
  if (!check_protected3a ())
    {
      puts ("`protected3' in moda has the wrong value");
      res = 1;
    }

  /* Check if we get the updated value for the protected data symbol.  */
  if (protected3 != 100)
    {
      puts ("`protected3' in main doesn't have the updated value");
      res = 1;
    }

  /* Check `protected3' in modb.  */
  if (!check_protected3b ())
    {
      puts ("`protected3' in modb has the wrong value");
      res = 1;
    }

  return res;
}
