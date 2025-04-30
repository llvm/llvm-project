/* Support functions related to (pseudo)terminals.
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

#ifndef _SUPPORT_TTY_H
#define _SUPPORT_TTY_H 1

struct termios;
struct winsize;

/** Open a pseudoterminal pair.  The outer fd is written to the address
    A_OUTER and the inner fd to A_INNER.

    If A_NAME is not NULL, it will be set to point to a string naming
    the /dev/pts/NNN device corresponding to the inner fd; space for
    this string is allocated with malloc and should be freed by the
    caller when no longer needed.  (This is different from the libutil
    function 'openpty'.)

    If TERMP is not NULL, the terminal parameters will be initialized
    according to the termios structure it points to.

    If WINP is not NULL, the terminal window size will be set
    accordingly.

    Terminates the process on failure (like xmalloc).  */
extern void support_openpty (int *a_outer, int *a_inner, char **a_name,
                             const struct termios *termp,
                             const struct winsize *winp);

#endif
