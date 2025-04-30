/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2000.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; see the file COPYING.LIB.  If
   not, see <https://www.gnu.org/licenses/>.  */

#include <stdint.h>


typedef uint16_t gidx_t;


struct gconvcache_header
{
  uint32_t magic;
  gidx_t string_offset;
  gidx_t hash_offset;
  gidx_t hash_size;
  gidx_t module_offset;
  gidx_t otherconv_offset;
};

struct hash_entry
{
  gidx_t string_offset;
  gidx_t module_idx;
};

struct module_entry
{
  gidx_t canonname_offset;
  gidx_t fromdir_offset;
  gidx_t fromname_offset;
  gidx_t todir_offset;
  gidx_t toname_offset;
  gidx_t extra_offset;
};

struct extra_entry
{
  gidx_t module_cnt;
  struct extra_entry_module
  {
    gidx_t outname_offset;
    gidx_t dir_offset;
    gidx_t name_offset;
  } module[0];
};


#define GCONVCACHE_MAGIC	0x20010324


#define GCONV_MODULES_CACHE	GCONV_DIR "/gconv-modules.cache"
