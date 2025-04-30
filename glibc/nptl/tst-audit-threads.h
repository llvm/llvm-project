/* Helper header for test-audit-threads.

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

/* We use this helper to create a large number of functions, all of
   which will be resolved lazily and thus have their PLT updated.
   This is done to provide enough functions that we can statistically
   observe a thread vs. PLT resolution failure if one exists.  */

#define CONCAT(a, b) a ## b
#define NUM(x, y) CONCAT (x, y)

#define FUNC10(x)	\
  FUNC (NUM (x, 0));	\
  FUNC (NUM (x, 1));	\
  FUNC (NUM (x, 2));	\
  FUNC (NUM (x, 3));	\
  FUNC (NUM (x, 4));	\
  FUNC (NUM (x, 5));	\
  FUNC (NUM (x, 6));	\
  FUNC (NUM (x, 7));	\
  FUNC (NUM (x, 8));	\
  FUNC (NUM (x, 9))

#define FUNC100(x)	\
  FUNC10 (NUM (x, 0));	\
  FUNC10 (NUM (x, 1));	\
  FUNC10 (NUM (x, 2));	\
  FUNC10 (NUM (x, 3));	\
  FUNC10 (NUM (x, 4));	\
  FUNC10 (NUM (x, 5));	\
  FUNC10 (NUM (x, 6));	\
  FUNC10 (NUM (x, 7));	\
  FUNC10 (NUM (x, 8));	\
  FUNC10 (NUM (x, 9))

#define FUNC1000(x)		\
  FUNC100 (NUM (x, 0));		\
  FUNC100 (NUM (x, 1));		\
  FUNC100 (NUM (x, 2));		\
  FUNC100 (NUM (x, 3));		\
  FUNC100 (NUM (x, 4));		\
  FUNC100 (NUM (x, 5));		\
  FUNC100 (NUM (x, 6));		\
  FUNC100 (NUM (x, 7));		\
  FUNC100 (NUM (x, 8));		\
  FUNC100 (NUM (x, 9))

#define FUNC7000()	\
  FUNC1000 (1);		\
  FUNC1000 (2);		\
  FUNC1000 (3);		\
  FUNC1000 (4);		\
  FUNC1000 (5);		\
  FUNC1000 (6);		\
  FUNC1000 (7);

#ifdef FUNC
# undef FUNC
#endif

#ifdef externnum
# define FUNC(x) extern int CONCAT (retNum, x) (void)
#endif

#ifdef definenum
# define FUNC(x) int CONCAT (retNum, x) (void) { return x; }
#endif

#ifdef callnum
# define FUNC(x) CONCAT (retNum, x) (); sync_all (x)
#endif

/* A value of 7000 functions is chosen as an arbitrarily large
   number of functions that will allow us enough attempts to
   verify lazy resolution operation.  */
FUNC7000 ();
