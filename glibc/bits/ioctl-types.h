/* Structure types for pre-termios terminal ioctls.  Generic Unix version.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
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

#ifndef _SYS_IOCTL_H
# error "Never use <bits/ioctl-types.h> directly; include <sys/ioctl.h> instead."
#endif

#if defined TIOCGETC || defined TIOCSETC
/* Type of ARG for TIOCGETC and TIOCSETC requests.  */
struct tchars
{
  char t_intrc;			/* Interrupt character.  */
  char t_quitc;			/* Quit character.  */
  char t_startc;		/* Start-output character.  */
  char t_stopc;			/* Stop-output character.  */
  char t_eofc;			/* End-of-file character.  */
  char t_brkc;			/* Input delimiter character.  */
};

# define	_IOT_tchars	/* Hurd ioctl type field.  */ \
  _IOT (_IOTS (char), 6, 0, 0, 0, 0)
#endif

#if defined TIOCGLTC || defined TIOCSLTC
/* Type of ARG for TIOCGLTC and TIOCSLTC requests.  */
struct ltchars
{
  char t_suspc;			/* Suspend character.  */
  char t_dsuspc;		/* Delayed suspend character.  */
  char t_rprntc;		/* Reprint-line character.  */
  char t_flushc;		/* Flush-output character.  */
  char t_werasc;		/* Word-erase character.  */
  char t_lnextc;		/* Literal-next character.  */
};

# define	_IOT_ltchars	/* Hurd ioctl type field.  */ \
  _IOT (_IOTS (char), 6, 0, 0, 0, 0)
#endif

/* Type of ARG for TIOCGETP and TIOCSETP requests (and gtty and stty).  */
struct sgttyb
{
  char sg_ispeed;		/* Input speed.  */
  char sg_ospeed;		/* Output speed.  */
  char sg_erase;		/* Erase character.  */
  char sg_kill;			/* Kill character.  */
  short int sg_flags;		/* Mode flags.  */
};

#define	_IOT_sgttyb	/* Hurd ioctl type field.  */ \
  _IOT (_IOTS (char), 4, _IOTS (short int), 1, 0, 0)

#if defined TIOCGWINSZ || defined TIOCSWINSZ
/* Type of ARG for TIOCGWINSZ and TIOCSWINSZ requests.  */
struct winsize
{
  unsigned short int ws_row;	/* Rows, in characters.  */
  unsigned short int ws_col;	/* Columns, in characters.  */

  /* These are not actually used.  */
  unsigned short int ws_xpixel;	/* Horizontal pixels.  */
  unsigned short int ws_ypixel;	/* Vertical pixels.  */
};

# define	_IOT_winsize	/* Hurd ioctl type field.  */ \
  _IOT (_IOTS (unsigned short int), 4, 0, 0, 0, 0)
#endif

#if defined TIOCGSIZE || defined TIOCSSIZE
/* The BSD-style ioctl constructor macros use `sizeof', which can't be used
   in a preprocessor conditional.  Since the commands are always unique
   regardless of the size bits, we can safely define away `sizeof' for the
   purpose of the conditional.  */
# define sizeof(type) 0
# if defined TIOCGWINSZ && TIOCGSIZE == TIOCGWINSZ
/* Many systems that have TIOCGWINSZ define TIOCGSIZE for source
   compatibility with Sun; they define `struct ttysize' to have identical
   layout as `struct winsize' and #define TIOCGSIZE to be TIOCGWINSZ
   (likewise TIOCSSIZE and TIOCSWINSZ).  */
struct ttysize
{
  unsigned short int ts_lines;
  unsigned short int ts_cols;
  unsigned short int ts_xxx;
  unsigned short int ts_yyy;
};
#  define	_IOT_ttysize	_IOT_winsize
# else
/* Suns use a different layout for `struct ttysize', and TIOCGSIZE and
   TIOCGWINSZ are separate commands that do the same thing with different
   structures (likewise TIOCSSIZE and TIOCSWINSZ).  */
struct ttysize
{
  int ts_lines, ts_cols;	/* Lines and columns, in characters.  */
};
# endif
# undef sizeof			/* See above.  */
#endif
