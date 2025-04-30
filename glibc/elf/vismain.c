/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
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

#include "vismod.h"

/* Prototype for our test function.  */
extern int do_test (void);


/* This defines the `main' function and some more.  */
#include <support/test-driver.c>


/* Prototypes for local functions.  */
extern int protlocal (void);

const char *protvarlocal = __FILE__;
extern const char *protvarinmod;
extern const char *protvaritcpt;

int
do_test (void)
{
  int res = 0;
  int val;

  /* First test: check whether .protected is handled correctly by the
     assembler/linker.  The uses of `protlocal' in the DSOs and in the
     main program should all be resolved with the local definitions.  */
  val = protlocal () + calllocal1 () + calllocal2 ();
  if (val != 0x155)
    {
      puts ("\
The handling of `.protected' seems to be implemented incorrectly: giving up");
      abort ();
    }
  puts ("`.protected' seems to be handled correctly, good!");

  /* Function pointers: for functions which are marked local and for
     which definitions are available all function pointers must be
     distinct.  */
  if (protlocal == getlocal1 ())
    {
      puts ("`protlocal' in main and mod1 have same address");
      res = 1;
    }
  if (protlocal == getlocal2 ())
    {
      puts ("`protlocal' in main and mod2 have same address");
      res = 1;
    }
  if (getlocal1 () == getlocal2 ())
    {
      puts ("`protlocal' in mod1 and mod2 have same address");
      res = 1;
    }
  if (getlocal1 () () + getlocal2 () () != 0x44)
    {
      puts ("pointers to `protlocal' in mod1 or mod2 incorrect");
      res = 1;
    }

  /* Next test.  This is similar to the last one but the function we
     are calling is not defined in the main object.  This means that
     the invocation in the main object uses the definition in the
     first DSO.  */
  if (protinmod != getinmod1 ())
    {
      printf ("&protinmod in main (%p) != &protinmod in mod1 (%p)\n",
	      protinmod, getinmod1 ());
      res = 1;
    }
  if (protinmod == getinmod2 ())
    {
      puts ("`protinmod' in main and mod2 have same address");
      res = 1;
    }
  if (getinmod1 () == getinmod2 ())
    {
      puts ("`protinmod' in mod1 and mod2 have same address");
      res = 1;
    }
  if (protinmod () + getinmod1 () () + getinmod2 () () != 0x4800)
    {
      puts ("pointers to `protinmod' in mod1 or mod2 incorrect");
      res = 1;
    }
  val = protinmod () + callinmod1 () + callinmod2 ();
  if (val != 0x15800)
    {
      printf ("calling of `protinmod' leads to wrong result (%#x)\n", val);
      res = 1;
    }

  /* A very similar text.  Same setup for the main object and the modules
     but this time we have another definition in a preloaded module. This
     one intercepts the references from the main object.  */
  if (protitcpt != getitcpt3 ())
    {
      printf ("&protitcpt in main (%p) != &protitcpt in mod3 (%p)\n",
	      &protitcpt, getitcpt3 ());
      res = 1;
    }
  if (protitcpt == getitcpt1 ())
    {
      puts ("`protitcpt' in main and mod1 have same address");
      res = 1;
    }
  if (protitcpt == getitcpt2 ())
    {
      puts ("`protitcpt' in main and mod2 have same address");
      res = 1;
    }
  if (getitcpt1 () == getitcpt2 ())
    {
      puts ("`protitcpt' in mod1 and mod2 have same address");
      res = 1;
    }
  val = protitcpt () + getitcpt1 () () + getitcpt2 () () + getitcpt3 () ();
  if (val != 0x8440000)
    {
      printf ("\
pointers to `protitcpt' in mod1 or mod2 or mod3 incorrect (%#x)\n", val);
      res = 1;
    }
  val = protitcpt () + callitcpt1 () + callitcpt2 () + callitcpt3 ();
  if (val != 0x19540000)
    {
      printf ("calling of `protitcpt' leads to wrong result (%#x)\n", val);
      res = 1;
    }

  /* Now look at variables.  First a variable which is available
     everywhere.  We must have three different addresses.  */
  if (&protvarlocal == getvarlocal1 ())
    {
      puts ("`protvarlocal' in main and mod1 have same address");
      res = 1;
    }
  if (&protvarlocal == getvarlocal2 ())
    {
      puts ("`protvarlocal' in main and mod2 have same address");
      res = 1;
    }
  if (getvarlocal1 () == getvarlocal2 ())
    {
      puts ("`protvarlocal' in mod1 and mod2 have same address");
      res = 1;
    }
  if (strcmp (protvarlocal, __FILE__) != 0)
    {
      puts ("`protvarlocal in main has wrong value");
      res = 1;
    }
  if (strcmp (*getvarlocal1 (), "vismod1.c") != 0)
    {
      puts ("`getvarlocal1' returns wrong value");
      res = 1;
    }
  if (strcmp (*getvarlocal2 (), "vismod2.c") != 0)
    {
      puts ("`getvarlocal2' returns wrong value");
      res = 1;
    }

  /* Now the case where there is no local definition.  */
  if (&protvarinmod != getvarinmod1 ())
    {
      printf ("&protvarinmod in main (%p) != &protitcpt in mod1 (%p)\n",
	      &protvarinmod, getvarinmod1 ());
      // XXX Possibly enable once fixed.
      // res = 1;
    }
  if (&protvarinmod == getvarinmod2 ())
    {
      puts ("`protvarinmod' in main and mod2 have same address");
      res = 1;
    }
  if (strcmp (*getvarinmod1 (), "vismod1.c") != 0)
    {
      puts ("`getvarinmod1' returns wrong value");
      res = 1;
    }
  if (strcmp (*getvarinmod2 (), "vismod2.c") != 0)
    {
      puts ("`getvarinmod2' returns wrong value");
      res = 1;
    }

  /* And a test where a variable definition is intercepted.  */
  if (&protvaritcpt == getvaritcpt1 ())
    {
      puts ("`protvaritcpt' in main and mod1 have same address");
      res = 1;
    }
  if (&protvaritcpt == getvaritcpt2 ())
    {
      puts ("`protvaritcpt' in main and mod2 have same address");
      res = 1;
    }
  if (&protvaritcpt != getvaritcpt3 ())
    {
      printf ("&protvaritcpt in main (%p) != &protvaritcpt in mod3 (%p)\n",
	      &protvaritcpt, getvaritcpt3 ());
      // XXX Possibly enable once fixed.
      // res = 1;
    }
  if (getvaritcpt1 () == getvaritcpt2 ())
    {
      puts ("`protvaritcpt' in mod1 and mod2 have same address");
      res = 1;
    }
  if (strcmp (protvaritcpt, "vismod3.c") != 0)
    {
      puts ("`protvaritcpt in main has wrong value");
      res = 1;
    }
  if (strcmp (*getvaritcpt1 (), "vismod1.c") != 0)
    {
      puts ("`getvaritcpt1' returns wrong value");
      res = 1;
    }
  if (strcmp (*getvaritcpt2 (), "vismod2.c") != 0)
    {
      puts ("`getvaritcpt2' returns wrong value");
      res = 1;
    }

  return res;
}


int
protlocal (void)
{
  return 0x1;
}
