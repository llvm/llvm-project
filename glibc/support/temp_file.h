/* Declarations for temporary file handling.
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

#ifndef SUPPORT_TEMP_FILE_H
#define SUPPORT_TEMP_FILE_H

#include <sys/cdefs.h>

__BEGIN_DECLS

/* Schedule a temporary file for deletion on exit.  */
void add_temp_file (const char *name);

/* Create a temporary file.  Return the opened file descriptor on
   success, or -1 on failure.  Write the file name to *FILENAME if
   FILENAME is not NULL.  In this case, the caller is expected to free
   *FILENAME.  */
int create_temp_file (const char *base, char **filename);

/* Create a temporary file in directory DIR.  Return the opened file
   descriptor on success, or -1 on failure.  Write the file name to
   *FILENAME if FILENAME is not NULL.  In this case, the caller is
   expected to free *FILENAME.  */
int create_temp_file_in_dir (const char *base, const char *dir,
			     char **filename);

/* Create a temporary directory and schedule it for deletion.  BASE is
   used as a prefix for the unique directory name, which the function
   returns.  The caller should free this string.  */
char *support_create_temp_directory (const char *base);

__END_DECLS

#endif /* SUPPORT_TEMP_FILE_H */
