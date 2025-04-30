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

#include <stdlib.h>
#include "vismod.h"

int
protlocal (void)
{
  return 0x40;
}
asm (".protected protlocal");


int
calllocal2 (void)
{
  return protlocal () + 0x100;
}

int
(*getlocal2 (void)) (void)
{
  return protlocal;
}

int
protinmod (void)
{
  return 0x4000;
}
asm (".protected protinmod");

int
callinmod2 (void)
{
  return protinmod () + 0x10000;
}

int
(*getinmod2 (void)) (void)
{
  return protinmod;
}

int
protitcpt (void)
{
  return 0x400000;
}
asm (".protected protitcpt");

int
callitcpt2 (void)
{
  return protitcpt () + 0x1000000;
}

int
(*getitcpt2 (void)) (void)
{
  return protitcpt;
}

const char *protvarlocal = __FILE__;
asm (".protected protvarlocal");

const char **
getvarlocal2 (void)
{
  return &protvarlocal;
}

const char *protvarinmod = __FILE__;
asm (".protected protvarinmod");

const char **
getvarinmod2 (void)
{
  return &protvarinmod;
}

const char *protvaritcpt = __FILE__;
asm (".protected protvaritcpt");

const char **
getvaritcpt2 (void)
{
  return &protvaritcpt;
}

/* We must never call these functions.  */
int
callitcpt3 (void)
{
  abort ();
}

int
(*getitcpt3 (void)) (void)
{
  abort ();
}

const char **
getvaritcpt3 (void)
{
  abort ();
}
