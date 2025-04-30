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

#include "vismod.h"

int
protlocal (void)
{
  return 0x4;
}
asm (".protected protlocal");


int
calllocal1 (void)
{
  return protlocal () + 0x10;
}

int
(*getlocal1 (void)) (void)
{
  return protlocal;
}

int
protinmod (void)
{
  return 0x400;
}
asm (".protected protinmod");

int
callinmod1 (void)
{
  return protinmod () + 0x1000;
}

int
(*getinmod1 (void)) (void)
{
  return protinmod;
}

int
protitcpt (void)
{
  return 0x40000;
}
asm (".protected protitcpt");

int
callitcpt1 (void)
{
  return protitcpt () + 0x100000;
}

int
(*getitcpt1 (void)) (void)
{
  return protitcpt;
}

const char *protvarlocal = __FILE__;
asm (".protected protvarlocal");

const char **
getvarlocal1 (void)
{
  return &protvarlocal;
}

const char *protvarinmod = __FILE__;
asm (".protected protvarinmod");

const char **
getvarinmod1 (void)
{
  return &protvarinmod;
}

const char *protvaritcpt = __FILE__;
asm (".protected protvaritcpt");

const char **
getvaritcpt1 (void)
{
  return &protvaritcpt;
}
