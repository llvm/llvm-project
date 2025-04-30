/* Interfaces for printing diagnostics in ld.so.
   Copyright (C) 2021 Free Software Foundation, Inc.
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

#ifndef _DL_DIAGNOSTICS_H
#define _DL_DIAGNOSTICS_H

#include <stdint.h>

/* Write the null-terminated string to standard output, surrounded in
   quotation marks.  */
void _dl_diagnostics_print_string (const char *s) attribute_hidden;

/* Like _dl_diagnostics_print_string, but add a LABEL= prefix, and a
   newline character as a suffix.  */
void _dl_diagnostics_print_labeled_string (const char *label, const char *s)
  attribute_hidden;

/* Print LABEL=VALUE to standard output, followed by a newline
   character.  */
void _dl_diagnostics_print_labeled_value (const char *label, uint64_t value)
  attribute_hidden;

/* Print diagnostics data for the kernel.  Called from
   _dl_print_diagnostics.  */
void _dl_diagnostics_kernel (void) attribute_hidden;

/* Print diagnostics data for the CPU(s).  Called from
   _dl_print_diagnostics.  */
void _dl_diagnostics_cpu (void) attribute_hidden;

#endif /* _DL_DIAGNOSTICS_H */
