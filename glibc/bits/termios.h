/* termios type and macro definitions.  4.4 BSD/generic GNU version.
   Copyright (C) 1993-2021 Free Software Foundation, Inc.
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

#ifndef _TERMIOS_H
# error "Never include <bits/termios.h> directly; use <termios.h> instead."
#endif

/* These macros are also defined in some <bits/ioctls.h> files (with
   numerically identical values), but this serves to shut up cpp's
   complaining. */
#if defined __USE_MISC || defined __USE_XOPEN

# ifdef NL0
#  undef NL0
# endif
# ifdef NL1
#  undef NL1
# endif
# ifdef TAB0
#  undef TAB0
# endif
# ifdef TAB1
#  undef TAB1
# endif
# ifdef TAB2
#  undef TAB2
# endif
# ifdef CR0
#  undef CR0
# endif
# ifdef CR1
#  undef CR1
# endif
# ifdef CR2
#  undef CR2
# endif
# ifdef CR3
#  undef CR3
# endif
# ifdef FF0
#  undef FF0
# endif
# ifdef FF1
#  undef FF1
# endif
# ifdef BS0
#  undef BS0
# endif
# ifdef BS1
#  undef BS1
# endif

#endif /* __USE_MISC || __USE_XOPEN */

#ifdef __USE_MISC

# ifdef MDMBUF
#  undef MDMBUF
# endif
# ifdef FLUSHO
#  undef FLUSHO
# endif
# ifdef PENDIN
#  undef PENDIN
# endif

#endif /* __USE_MISC */

#ifdef ECHO
# undef ECHO
#endif
#ifdef TOSTOP
# undef TOSTOP
#endif
#ifdef NOFLSH
# undef NOFLSH
#endif


/* These definitions match those used by the 4.4 BSD kernel.
   If the operating system has termios system calls or ioctls that
   correctly implement the POSIX.1 behavior, there should be a
   system-dependent version of this file that defines `struct termios',
   `tcflag_t', `cc_t', `speed_t' and the `TC*' constants appropriately.  */

/* Type of terminal control flag masks.  */
typedef unsigned long int tcflag_t;

/* Type of control characters.  */
typedef unsigned char cc_t;

/* Type of baud rate specifiers.  */
typedef long int speed_t;

/* Terminal control structure.  */
struct termios
{
  /* Input modes.  */
  tcflag_t c_iflag;
#define	IGNBRK	(1 << 0)	/* Ignore break condition.  */
#define	BRKINT	(1 << 1)	/* Signal interrupt on break.  */
#define	IGNPAR	(1 << 2)	/* Ignore characters with parity errors.  */
#define	PARMRK	(1 << 3)	/* Mark parity and framing errors.  */
#define	INPCK	(1 << 4)	/* Enable input parity check.  */
#define	ISTRIP	(1 << 5)	/* Strip 8th bit off characters.  */
#define	INLCR	(1 << 6)	/* Map NL to CR on input.  */
#define	IGNCR	(1 << 7)	/* Ignore CR.  */
#define	ICRNL	(1 << 8)	/* Map CR to NL on input.  */
#define	IXON	(1 << 9)	/* Enable start/stop output control.  */
#define	IXOFF	(1 << 10)	/* Enable start/stop input control.  */
#if defined __USE_MISC || defined __USE_XOPEN || defined __USE_XOPEN2K8
# define IXANY	(1 << 11)	/* Any character will restart after stop.  */
#endif
#ifdef	__USE_MISC
# define IMAXBEL (1 << 13)	/* Ring bell when input queue is full.  */
#endif
#if defined __USE_GNU || (defined __USE_XOPEN && !defined __USE_XOPEN2K)
# define IUCLC	(1 << 14)	/* Translate upper case input to lower case. */
#endif

  /* Output modes.  */
  tcflag_t c_oflag;
#define	OPOST	(1 << 0)	/* Perform output processing.  */
#if defined __USE_MISC || defined __USE_XOPEN
# define ONLCR	(1 << 1)	/* Map NL to CR-NL on output.  */
#endif
#ifdef	__USE_MISC
# define OXTABS	TAB3		/* Expand tabs to spaces.  */
# define ONOEOT	(1 << 3)	/* Discard EOT (^D) on output.  */
#endif
#if defined __USE_MISC || defined __USE_XOPEN
# define OCRNL	(1 << 4)	/* Map CR to NL.  */
# define ONOCR	(1 << 5)	/* Discard CR's when on column 0.  */
# define ONLRET	(1 << 6)	/* Move to column 0 on NL.  */
#endif
#if defined __USE_MISC || defined __USE_XOPEN
# define NLDLY	(3 << 8)	/* NL delay.  */
# define NL0	(0 << 8)	/* NL type 0.  */
# define NL1	(1 << 8)	/* NL type 1.  */
# define TABDLY	(3 << 10 | 1 << 2)	/* TAB delay.  */
# define TAB0	(0 << 10)	/* TAB delay type 0.  */
# define TAB1	(1 << 10)	/* TAB delay type 1.  */
# define TAB2	(2 << 10)	/* TAB delay type 2.  */
# define TAB3	(1 << 2)	/* Expand tabs to spaces.  */
# define CRDLY	(3 << 12)	/* CR delay.  */
# define CR0	(0 << 12)	/* CR delay type 0.  */
# define CR1	(1 << 12)	/* CR delay type 1.  */
# define CR2	(2 << 12)	/* CR delay type 2.  */
# define CR3	(3 << 12)	/* CR delay type 3.  */
# define FFDLY	(1 << 14)	/* FF delay.  */
# define FF0	(0 << 14)	/* FF delay type 0.  */
# define FF1	(1 << 14)	/* FF delay type 1.  */
# define BSDLY	(1 << 15)	/* BS delay.  */
# define BS0	(0 << 15)	/* BS delay type 0.  */
# define BS1	(1 << 15)	/* BS delay type 1.  */
# define VTDLY	(1 << 16)	/* VT delay.  */
# define VT0	(0 << 16)	/* VT delay type 0.  */
# define VT1	(1 << 16)	/* VT delay type 1.  */
#endif /* __USE_MISC || __USE_XOPEN */
#if defined __USE_GNU || (defined __USE_XOPEN && !defined __USE_XOPEN2K)
# define OLCUC	(1 << 17)	/* Translate lower case output to upper case */
#endif
#ifdef __USE_XOPEN
# define OFILL	(1 << 18)	/* Send fill characters for delays.  */
# define OFDEL	(1 << 19)	/* Fill is DEL.  */
#endif

  /* Control modes.  */
  tcflag_t c_cflag;
#ifdef	__USE_MISC
# define CIGNORE	(1 << 0)	/* Ignore these control flags.  */
#endif
#define	CSIZE	(CS5|CS6|CS7|CS8)	/* Number of bits per byte (mask).  */
#define	CS5	0		/* 5 bits per byte.  */
#define	CS6	(1 << 8)	/* 6 bits per byte.  */
#define	CS7	(1 << 9)	/* 7 bits per byte.  */
#define	CS8	(CS6|CS7)	/* 8 bits per byte.  */
#define	CSTOPB	(1 << 10)	/* Two stop bits instead of one.  */
#define	CREAD	(1 << 11)	/* Enable receiver.  */
#define	PARENB	(1 << 12)	/* Parity enable.  */
#define	PARODD	(1 << 13)	/* Odd parity instead of even.  */
#define	HUPCL	(1 << 14)	/* Hang up on last close.  */
#define	CLOCAL	(1 << 15)	/* Ignore modem status lines.  */
#ifdef	__USE_MISC
# define CRTSCTS	(1 << 16)	/* RTS/CTS flow control.  */
# define CRTS_IFLOW	CRTSCTS		/* Compatibility.  */
# define CCTS_OFLOW	CRTSCTS		/* Compatibility.  */
# define CDTRCTS	(1 << 17)	/* DTR/CTS flow control.  */
# define MDMBUF		(1 << 20)	/* DTR/DCD flow control.  */
# define CHWFLOW	(MDMBUF|CRTSCTS|CDTRCTS) /* All types of flow control.  */
#endif

  /* Local modes.  */
  tcflag_t c_lflag;
#ifdef	__USE_MISC
# define ECHOKE	(1 << 0)	/* Visual erase for KILL.  */
#endif
#define	_ECHOE	(1 << 1)	/* Visual erase for ERASE.  */
#define	ECHOE	_ECHOE
#define	_ECHOK	(1 << 2)	/* Echo NL after KILL.  */
#define	ECHOK	_ECHOK
#define	_ECHO	(1 << 3)	/* Enable echo.  */
#define	ECHO	_ECHO
#define	_ECHONL	(1 << 4)	/* Echo NL even if ECHO is off.  */
#define	ECHONL	_ECHONL
#ifdef	__USE_MISC
# define ECHOPRT	(1 << 5)	/* Hardcopy visual erase.  */
# define ECHOCTL	(1 << 6)	/* Echo control characters as ^X.  */
#endif
#define	_ISIG	(1 << 7)	/* Enable signals.  */
#define	ISIG	_ISIG
#define	_ICANON	(1 << 8)	/* Do erase and kill processing.  */
#define	ICANON	_ICANON
#ifdef	__USE_MISC
# define ALTWERASE (1 << 9)	/* Alternate WERASE algorithm.  */
#endif
#define	_IEXTEN	(1 << 10)	/* Enable DISCARD and LNEXT.  */
#define	IEXTEN	_IEXTEN
#ifdef	__USE_MISC
# define EXTPROC	(1 << 11)	/* External processing.  */
#endif
#define	_TOSTOP	(1 << 22)	/* Send SIGTTOU for background output.  */
#define	TOSTOP	_TOSTOP
#ifdef	__USE_MISC
# define FLUSHO	(1 << 23)	/* Output being flushed (state).  */
#endif
#if defined __USE_XOPEN && !defined __USE_XOPEN2K
# define XCASE	(1 << 24)	/* Canonical upper/lower case.  */
#endif
#ifdef __USE_MISC
# define NOKERNINFO (1 << 25)	/* Disable VSTATUS.  */
# define PENDIN	(1 << 29)	/* Retype pending input (state).  */
#endif
#define	_NOFLSH	(1 << 31)	/* Disable flush after interrupt.  */
#define	NOFLSH	_NOFLSH

  /* Control characters.  */
#define	VEOF	0		/* End-of-file character [ICANON].  */
#define	VEOL	1		/* End-of-line character [ICANON].  */
#ifdef	__USE_MISC
# define VEOL2	2		/* Second EOL character [ICANON].  */
#endif
#define	VERASE	3		/* Erase character [ICANON].  */
#ifdef	__USE_MISC
# define VWERASE	4		/* Word-erase character [ICANON].  */
#endif
#define	VKILL	5		/* Kill-line character [ICANON].  */
#ifdef	__USE_MISC
# define VREPRINT 6		/* Reprint-line character [ICANON].  */
#endif
#define	VINTR	8		/* Interrupt character [ISIG].  */
#define	VQUIT	9		/* Quit character [ISIG].  */
#define	VSUSP	10		/* Suspend character [ISIG].  */
#ifdef	__USE_MISC
# define VDSUSP	11		/* Delayed suspend character [ISIG].  */
#endif
#define	VSTART	12		/* Start (X-ON) character [IXON, IXOFF].  */
#define	VSTOP	13		/* Stop (X-OFF) character [IXON, IXOFF].  */
#ifdef	__USE_MISC
# define VLNEXT	14		/* Literal-next character [IEXTEN].  */
# define VDISCARD 15		/* Discard character [IEXTEN].  */
#endif
#define	VMIN	16		/* Minimum number of bytes read at once [!ICANON].  */
#define	VTIME	17		/* Time-out value (tenths of a second) [!ICANON].  */
#ifdef	__USE_MISC
# define VSTATUS	18		/* Status character [ICANON].  */
#endif
#define	NCCS	20		/* Value duplicated in <hurd/tioctl.defs>.  */
  cc_t c_cc[NCCS];

  /* Input and output baud rates.  */
  speed_t __ispeed, __ospeed;
#define	B0	0		/* Hang up.  */
#define	B50	50		/* 50 baud.  */
#define	B75	75		/* 75 baud.  */
#define	B110	110		/* 110 baud.  */
#define	B134	134		/* 134.5 baud.  */
#define	B150	150		/* 150 baud.  */
#define	B200	200		/* 200 baud.  */
#define	B300	300		/* 300 baud.  */
#define	B600	600		/* 600 baud.  */
#define	B1200	1200		/* 1200 baud.  */
#define	B1800	1800		/* 1800 baud.  */
#define	B2400	2400		/* 2400 baud.  */
#define	B4800	4800		/* 4800 baud.  */
#define	B9600	9600		/* 9600 baud.  */
#define	B7200	7200		/* 7200 baud.  */
#define	B14400	14400		/* 14400 baud.  */
#define	B19200	19200		/* 19200 baud.  */
#define	B28800	28800		/* 28800 baud.  */
#define	B38400	38400		/* 38400 baud.  */
#ifdef	__USE_MISC
# define EXTA	19200
# define EXTB	38400
#endif
#define	B57600	57600
#define	B76800	76800
#define	B115200	115200
#define	B230400	230400
#define	B460800	460800
#define	B500000	500000
#define	B576000	576000
#define	B921600	921600
#define	B1000000 1000000
#define	B1152000 1152000
#define	B1500000 1500000
#define	B2000000 2000000
#define	B2500000 2500000
#define	B3000000 3000000
#define	B3500000 3500000
#define	B4000000 4000000
};

#define _IOT_termios /* Hurd ioctl type field.  */ \
  _IOT (_IOTS (tcflag_t), 4, _IOTS (cc_t), NCCS, _IOTS (speed_t), 2)

/* Values for the OPTIONAL_ACTIONS argument to `tcsetattr'.  */
#define	TCSANOW		0	/* Change immediately.  */
#define	TCSADRAIN	1	/* Change when pending output is written.  */
#define	TCSAFLUSH	2	/* Flush pending input before changing.  */
#ifdef	__USE_MISC
# define TCSASOFT	0x10	/* Flag: Don't alter hardware state.  */
#endif

/* Values for the QUEUE_SELECTOR argument to `tcflush'.  */
#define	TCIFLUSH	1	/* Discard data received but not yet read.  */
#define	TCOFLUSH	2	/* Discard data written but not yet sent.  */
#define	TCIOFLUSH	3	/* Discard all pending data.  */

/* Values for the ACTION argument to `tcflow'.  */
#define	TCOOFF	1		/* Suspend output.  */
#define	TCOON	2		/* Restart suspended output.  */
#define	TCIOFF	3		/* Send a STOP character.  */
#define	TCION	4		/* Send a START character.  */
