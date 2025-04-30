/* Check that a versioned symbol can move from one library to another.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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

/* At link time, moved_function is bound to the symbol version
   SONAME_MOVE in tst-sonamemove-runmod1.so, using the
   tst-sonamemove-linkmod1.so stub object.

   At run time, the process loads the real tst-sonamemove-runmod1.so,
   which depends on tst-sonamemove-runmod2.so.
   tst-sonamemove-runmod1.so does not define moved_function, but
   tst-sonamemove-runmod2.so does.

   The net effect is that the versioned symbol
   moved_function@SONAME_MOVE moved from the soname
   tst-sonamemove-linkmod1.so at link time to the soname
   tst-sonamemove-linkmod2.so at run time. */
void moved_function (void);

static int
do_test (void)
{
  moved_function ();
  return 0;
}

#include <support/test-driver.c>
