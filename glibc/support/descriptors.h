/* Monitoring file descriptor usage.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#ifndef SUPPORT_DESCRIPTORS_H
#define SUPPORT_DESCRIPTORS_H

#include <stdio.h>

/* Opaque pointer, for capturing file descriptor lists.  */
struct support_descriptors;

/* Record the currently open file descriptors and store them in the
   returned list.  Terminate the process if the listing operation
   fails.  */
struct support_descriptors *support_descriptors_list (void);

/* Deallocate the list of descriptors.  */
void support_descriptors_free (struct support_descriptors *);

/* Write the list of descriptors to STREAM, adding PREFIX to each
   line.  */
void support_descriptors_dump (struct support_descriptors *,
                               const char *prefix, FILE *stream);

/* Check for file descriptor leaks and other file descriptor changes:
   Compare the current list of descriptors with the passed list.
   Record a test failure if there are additional open descriptors,
   descriptors have been closed, or if a change in file descriptor can
   be detected.  */
void support_descriptors_check (struct support_descriptors *);

#endif /* SUPPORT_DESCRIPTORS_H */
