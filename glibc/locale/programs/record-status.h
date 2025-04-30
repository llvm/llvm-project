/* General definitions for recording error and warning status.
   Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published
   by the Free Software Foundation; version 2 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, see <https://www.gnu.org/licenses/>.  */

#ifndef _RECORD_STATUS_H
#define _RECORD_STATUS_H 1

#include <stdio.h>
#include <stdbool.h>

/* Error, warning and verbose count and control.  */
extern int recorded_warning_count;
extern int recorded_error_count;
extern int be_quiet;
extern int verbose;
extern bool warn_ascii;
extern bool warn_int_curr_symbol;

/* Record verbose, warnings, or errors... */
void record_verbose (FILE *stream, const char *format, ...);
void record_warning (const char *format, ...);
void record_error (int status, int errnum, const char *format, ...);
void record_error_at_line (int status, int errnum,
			   const char *filename, unsigned int linenum,
			   const char *format, ...);

/* Locale related functionality for custom error functions.  */
struct locale_state
{
   /* The current in-use locale.  */
   char *cur_locale;
};

struct locale_state push_locale (void);
void pop_locale (struct locale_state ls);


#endif
