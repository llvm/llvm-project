/* Internal representation of tunables.

   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#ifndef _TUNABLE_TYPES_H_
#define _TUNABLE_TYPES_H_

/* Note: This header is included in the generated dl-tunables-list.h and
   only used internally in the tunables implementation in dl-tunables.c.  */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef enum
{
  TUNABLE_TYPE_INT_32,
  TUNABLE_TYPE_UINT_64,
  TUNABLE_TYPE_SIZE_T,
  TUNABLE_TYPE_STRING
} tunable_type_code_t;

typedef struct
{
  tunable_type_code_t type_code;
  tunable_num_t min;
  tunable_num_t max;
} tunable_type_t;

/* Security level for tunables.  This decides what to do with individual
   tunables for AT_SECURE binaries.  */
typedef enum
{
  /* Erase the tunable for AT_SECURE binaries so that child processes don't
     read it.  */
  TUNABLE_SECLEVEL_SXID_ERASE = 0,
  /* Ignore the tunable for AT_SECURE binaries, but don't erase it, so that
     child processes can read it.  */
  TUNABLE_SECLEVEL_SXID_IGNORE = 1,
  /* Read the tunable.  */
  TUNABLE_SECLEVEL_NONE = 2,
} tunable_seclevel_t;

/* A tunable.  */
struct _tunable
{
  const char name[TUNABLE_NAME_MAX];	/* Internal name of the tunable.  */
  tunable_type_t type;			/* Data type of the tunable.  */
  tunable_val_t val;			/* The value.  */
  bool initialized;			/* Flag to indicate that the tunable is
					   initialized.  */
  tunable_seclevel_t security_level;	/* Specify the security level for the
					   tunable with respect to AT_SECURE
					   programs.  See description of
					   tunable_seclevel_t to see a
					   description of the values.

					   Note that even if the tunable is
					   read, it may not get used by the
					   target module if the value is
					   considered unsafe.  */
  /* Compatibility elements.  */
  const char env_alias[TUNABLE_ALIAS_MAX]; /* The compatibility environment
					   variable name.  */
};

typedef struct _tunable tunable_t;

static __always_inline bool
unsigned_tunable_type (tunable_type_code_t t)
{
  switch (t)
    {
    case TUNABLE_TYPE_INT_32:
      return false;
    case TUNABLE_TYPE_UINT_64:
    case TUNABLE_TYPE_SIZE_T:
      return true;
    case TUNABLE_TYPE_STRING:
    default:
      break;
    }
  __builtin_unreachable ();
}

#endif
