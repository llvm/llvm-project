/* Copyright (C) 1992-2021 Free Software Foundation, Inc.
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

#include <termios.h>

/* Set *T to indicate raw mode.  */
void
cfmakeraw (struct termios *t)
{
  t->c_iflag &= ~(IGNBRK|BRKINT|PARMRK|ISTRIP|INLCR|IGNCR|ICRNL|IXON);
  t->c_oflag &= ~OPOST;
  t->c_lflag &= ~(ECHO|ECHONL|ICANON|ISIG|IEXTEN);
  t->c_cflag &= ~(CSIZE|PARENB);
  t->c_cflag |= CS8;
  t->c_cc[VMIN] = 1;		/* read returns when one char is available.  */
  t->c_cc[VTIME] = 0;
}
