/* Invoke the system diff tool to compare two strings.
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

#ifndef SUPPORT_RUN_DIFF_H
#define SUPPORT_RUN_DIFF_H

/* Compare the two NUL-terminated strings LEFT and RIGHT using the
   diff tool.  Label the sides of the diff with LEFT_LABEL and
   RIGHT_LABEL, respectively.

   This function assumes that LEFT and RIGHT are different
   strings.  */
void support_run_diff (const char *left_label, const char *left,
                       const char *right_label, const char *right);

#endif /* SUPPORT_RUN_DIFF_H */
