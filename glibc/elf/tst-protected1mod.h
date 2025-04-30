/* Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

/* Prototypes for the functions in the DSOs.  */
extern int protected1;
extern int protected2;
extern int protected3;

extern void set_protected1a (int);
extern void set_protected1b (int);
extern int *protected1a_p (void);
extern int *protected1b_p (void);

extern void set_expected_protected1 (int);
extern int check_protected1 (void);

extern void set_protected2 (int);
extern int check_protected2 (void);

extern void set_expected_protected3a (int);
extern void set_protected3a (int);
extern int check_protected3a (void);
extern int *protected3a_p (void);
extern void set_expected_protected3b (int);
extern void set_protected3b (int);
extern int check_protected3b (void);
extern int *protected3b_p (void);
