/* Internal weak declarations for temporary file handling.
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

#ifndef SUPPORT_TEMP_FILE_INTERNAL_H
#define SUPPORT_TEMP_FILE_INTERNAL_H

/* These functions are called by the test driver if they are
   defined.  Tests should not call them directly.  */

#include <stdio.h>

void support_set_test_dir (const char *name) __attribute__ ((weak));
void support_delete_temp_files (void) __attribute__ ((weak));
void support_print_temp_files (FILE *) __attribute__ ((weak));

#endif /* SUPPORT_TEMP_FILE_INTERNAL_H */
