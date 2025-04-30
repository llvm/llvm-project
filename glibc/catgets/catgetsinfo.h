/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper, <drepper@gnu.ai.mit.edu>.

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

#include <sys/types.h>


struct catalog_obj
{
  uint32_t magic;
  uint32_t plane_size;
  uint32_t plane_depth;
  /* This is in fact two arrays in one: always a pair of name and
     pointer into the data area.  */
  uint32_t name_ptr[0];
};


/* This structure will be filled after loading the catalog.  */
typedef struct catalog_info
{
  enum { mmapped, malloced } status;

  size_t plane_size;
  size_t plane_depth;
  uint32_t *name_ptr;
  const char *strings;

  struct catalog_obj *file_ptr;
  size_t file_size;
} *__nl_catd;



/* The magic number to signal we really have a catalog file.  */
#define CATGETS_MAGIC 0x960408deU


/* Prototypes for helper functions.  */
extern int __open_catalog (const char *cat_name, const char *nlspath,
			   const char *env_var, __nl_catd __catalog);
libc_hidden_proto (__open_catalog)
