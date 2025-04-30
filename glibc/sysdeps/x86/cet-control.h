/* x86 CET tuning.
   This file is part of the GNU C Library.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.

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

#ifndef _CET_CONTROL_H
#define _CET_CONTROL_H

/* For each CET feature, IBT and SHSTK, valid control values.  */
enum dl_x86_cet_control
{
  /* Enable CET features based on ELF property note.  */
  cet_elf_property = 0,
  /* Always enable CET features.  */
  cet_always_on,
  /* Always disable CET features.  */
  cet_always_off,
  /* Enable CET features permissively.  */
  cet_permissive
};

struct dl_x86_feature_control
{
  enum dl_x86_cet_control ibt : 2;
  enum dl_x86_cet_control shstk : 2;
};

#endif /* cet-control.h */
