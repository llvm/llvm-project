/* Shared definitions for dynarray tests.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#include <stddef.h>

struct int_array
{
  int *array;
  size_t length;
};

#define DYNARRAY_STRUCT dynarray_int
#define DYNARRAY_ELEMENT int
#define DYNARRAY_PREFIX dynarray_int_
#define DYNARRAY_FINAL_TYPE struct int_array
#include <malloc/dynarray-skeleton.c>

struct str_array
{
  char **array;
  size_t length;
};

#define DYNARRAY_STRUCT dynarray_str
#define DYNARRAY_ELEMENT char *
#define DYNARRAY_ELEMENT_FREE(ptr) free (*ptr)
#define DYNARRAY_PREFIX dynarray_str_
#define DYNARRAY_FINAL_TYPE struct str_array
#include <malloc/dynarray-skeleton.c>

/* Check that *DYN is equivalent to its initial state.  */
#define CHECK_INIT_STATE(type, dyn)                             \
  ({                                                            \
    TEST_VERIFY_EXIT (!dynarray_##type##_has_failed (dyn));     \
    TEST_VERIFY_EXIT (dynarray_##type##_size (dyn) == 0);       \
    TEST_VERIFY_EXIT ((dyn)->u.dynarray_header.array            \
                      == (dyn)->scratch);                       \
    TEST_VERIFY_EXIT ((dyn)->u.dynarray_header.allocated > 0);  \
    (void) 0;                                                   \
  })

/* Check that *DYN behaves as if it is in its initial state.  */
#define CHECK_EMPTY(type, dyn)                                       \
  ({                                                                 \
    CHECK_INIT_STATE (type, (dyn));                                  \
    dynarray_##type##_free (dyn);                                    \
    CHECK_INIT_STATE (type, (dyn));                                  \
    dynarray_##type##_clear (dyn);                                   \
    CHECK_INIT_STATE (type, (dyn));                                  \
    dynarray_##type##_remove_last (dyn);                             \
    CHECK_INIT_STATE (type, (dyn));                                  \
    dynarray_##type##_mark_failed (dyn);                             \
    TEST_VERIFY_EXIT (dynarray_##type##_has_failed (dyn));           \
    dynarray_##type##_clear (dyn);                                   \
    TEST_VERIFY_EXIT (dynarray_##type##_has_failed (dyn));           \
    dynarray_##type##_remove_last (dyn);                             \
    TEST_VERIFY_EXIT (dynarray_##type##_has_failed (dyn));           \
    TEST_VERIFY_EXIT (dynarray_##type##_emplace (dyn) == NULL);      \
    dynarray_##type##_free (dyn);                                    \
    CHECK_INIT_STATE (type, (dyn));                                  \
    /* These functions should not assert.  */                        \
    dynarray_##type##_begin (dyn);                                   \
    dynarray_##type##_end (dyn);                                     \
    (void) 0;                                                        \
  })
