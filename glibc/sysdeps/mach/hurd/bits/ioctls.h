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

#ifndef __BITS_IOCTLS_H
#define __BITS_IOCTLS_H	1

#if !defined _HURD_IOCTL_H && !defined _SYS_IOCTL_H
# error "Never use <bits/ioctls.h> directly; include <hurd/ioctl.h> instead."
#endif

/* These macros are also defined in <bits/termios.h> (with numerically
   identical values) but this serves to shut up cpp's complaining. */

#ifdef NL0
# undef NL0
#endif
#ifdef NL1
# undef NL1
#endif
#ifdef TAB0
# undef TAB0
#endif
#ifdef TAB1
# undef TAB1
#endif
#ifdef TAB2
# undef TAB2
#endif
#ifdef CR0
# undef CR0
#endif
#ifdef CR1
# undef CR1
#endif
#ifdef CR2
# undef CR2
#endif
#ifdef CR3
# undef CR3
#endif
#ifdef FF0
# undef FF0
#endif
#ifdef FF1
# undef FF1
#endif
#ifdef BS0
# undef BS0
#endif
#ifdef BS1
# undef BS1
#endif
#ifdef MDMBUF
# undef MDMBUF
#endif
#ifdef ECHO
# undef ECHO
#endif
#ifdef TOSTOP
# undef TOSTOP
#endif
#ifdef FLUSHO
# undef FLUSHO
#endif
#ifdef PENDIN
# undef PENDIN
#endif
#ifdef NOFLSH
# undef NOFLSH
#endif

/* Hurd ioctl request are made up of several fields:

    10987654321098765432109876543210
    IOt0t1t2cc0c0cc1c1cc2ggggccccccc

     bits [31,30]: inout direction (enum __ioctl_dir)
     bits [29,11]: type encoding as follows; zero count indicates omitted datum
	  [29,28]: datum #0 type (enum __ioctl_datum)
	  [27,26]: datum #1 type (enum __ioctl_datum)
	  [24,25]: datum #2 type (enum __ioctl_datum)
	  [23,19]: datum #0 count	[0,31]
	  [18,14]: datum #1 count [0,31]
	  [13,11]: datum #2 count [0,3]
     bits [07,10]: group (letter - 'f': ['f','v'])
     bits [00,06]: command	[0,127]

   The following macros construct and dissect these fields.  */

enum __ioctl_dir
  {
    IOC_VOID = 0,		/* No parameters.  */
    IOC_OUT = 1,		/* Data is written into the user's buffer.  */
    IOC_IN = 2,			/* Data is read from the user's buffer.  */
    IOC_INOUT = (IOC_IN|IOC_OUT)
  };

enum __ioctl_datum { IOC_8, IOC_16, IOC_32, IOC_64 };

/* Construct an ioctl from constructed type plus other fields.  */
#define	_IOC(inout, group, num, type) \
  ((num) | ((((group) - 'f') | ((type) | (inout) << 19) << 4) << 7))

/* Dissect an ioctl into its component fields.  */
#define _IOC_INOUT(request)	(((unsigned int) (request) >> 30) & IOC_INOUT)
#define _IOC_GROUP(request)	('f' + (((unsigned int) (request) >> 7) & 0xf))
#define _IOC_COMMAND(request)	((unsigned int) (request) & 0x7f)
#define _IOC_TYPE(request)	(((unsigned int) (request) >> 11) & 0x7ffff)
#define _IOC_NOTYPE(request)	((unsigned int) (request) & 0x3ff)

/* Construct a type information field from
   the broken-out type and count fields.  */
#define	_IOT(t0, c0, t1, c1, t2, c2) \
  ((c2) | (((c1) | ((c0) | ((t2) | ((t1) | (t0) << 2) << 2) << 5) << 5) << 3))

/* Dissect a type information field into the type and count fields.  */
#define	_IOT_TYPE0(type)	(((unsigned int) (type) >> 17) & 3)
#define	_IOT_TYPE1(type)	(((unsigned int) (type) >> 15) & 3)
#define	_IOT_TYPE2(type)	(((unsigned int) (type) >> 13) & 3)
#define	_IOT_COUNT0(type)	(((unsigned int) (type) >> 8) & 0x1f)
#define	_IOT_COUNT1(type)	(((unsigned int) (type) >> 3) & 0x1f)
#define	_IOT_COUNT2(type)	(((unsigned int) (type) >> 0) & 7)

/* Construct an ioctl from all the broken-out fields.  */
#define	_IOCT(inout, group, num, t0, c0, t1, c1, t2, c2)		      \
  _IOC ((inout), (group), (num), _IOT ((t0), (c0), (t1), (c1), (t2), (c2)))

/* Construct an individual type field for TYPE.  */
#define _IOTS(type)	\
  (sizeof (type) == 8 ? IOC_64 : (enum __ioctl_datum) (sizeof (type) >> 1))

/* Construct a type information field for
   a single argument of the scalar TYPE.  */
#define	_IOT_SIMPLE(type)	_IOT (_IOTS (type), 1, 0, 0, 0, 0)

/* Basic C types.  */
#define	_IOT__IOTBASE_char	_IOT_SIMPLE (char)
#define	_IOT__IOTBASE_short	_IOT_SIMPLE (short)
#define	_IOT__IOTBASE_int	_IOT_SIMPLE (int)
#define	_IOT__IOTBASE_long	_IOT_SIMPLE (long)
#define	_IOT_char		_IOT_SIMPLE (char)
#define	_IOT_short		_IOT_SIMPLE (short)
#define	_IOT_int		_IOT_SIMPLE (int)
#define	_IOT_long		_IOT_SIMPLE (long)

#define	_IOT__IOTBASE_int8_t	_IOT_SIMPLE (int8_t)
#define	_IOT__IOTBASE_uint8_t	_IOT_SIMPLE (uint8_t)
#define	_IOT__IOTBASE_int16_t	_IOT_SIMPLE (int16_t)
#define	_IOT__IOTBASE_uint16_t	_IOT_SIMPLE (uint16_t)
#define	_IOT__IOTBASE_int32_t	_IOT_SIMPLE (int32_t)
#define	_IOT__IOTBASE_uint32_t	_IOT_SIMPLE (uint32_t)
#define	_IOT__IOTBASE_int64_t	_IOT_SIMPLE (int64_t)
#define	_IOT__IOTBASE_uint64_t	_IOT_SIMPLE (uint64_t)

#define	_IOT__IOTBASE_size_t	_IOT_SIMPLE (size_t)
#define	_IOT__IOTBASE_ssize_t	_IOT_SIMPLE (ssize_t)


/* Standard flavors of ioctls.
   _IOT_foobar is defined either in this file,
   or where struct foobar is defined.  */
#define	_IO(g, n)	_IOC (IOC_VOID, (g), (n), 0)
#define	_IOIW(g, n, t)	_IOC (IOC_VOID, (g), (n), _IOC_ENCODE_TYPE (t))
#define	_IOR(g, n, t)	_IOC (IOC_OUT, (g), (n), _IOC_ENCODE_TYPE (t))
#define	_IOW(g, n, t)	_IOC (IOC_IN, (g), (n), _IOC_ENCODE_TYPE (t))
#define	_IOWR(g, n, t)	_IOC (IOC_INOUT, (g), (n), _IOC_ENCODE_TYPE (t))

/* These macros do some preprocessor gymnastics to turn a TYPESPEC of
   `struct foobar' into the identifier `_IOT_foobar', which is generally
   defined using `_IOT' (above) in whatever file defines `struct foobar'.
   For a TYPESPEC that does not begin with `struct' produces a different
   identifier: `int' produces `_IOT__IOTBASE_int'.  These identifiers
   are defined for the basic C types above.  */
#define _IOC_ENCODE_TYPE(typespec)	_IOC_ENCODE_TYPE_1(_IOTBASE_##typespec)
#define _IOTBASE_struct
#define _IOC_ENCODE_TYPE_1(typespec)	_IOC_ENCODE_TYPE_2(typespec)
#define _IOC_ENCODE_TYPE_2(typespec)	_IOT_##typespec

/* Also, ignore signedness.  */
#define	_IOTBASE_unsigned
#define	_IOTBASE_signed


/* ioctls verbatim from 4.4 <sys/ioctl.h>.  */

#define	TIOCMODG	_IOR('t', 3, int)	/* get modem control state */
#define	TIOCMODS	_IOW('t', 4, int)	/* set modem control state */
#define		TIOCM_LE	0001		/* line enable */
#define		TIOCM_DTR	0002		/* data terminal ready */
#define		TIOCM_RTS	0004		/* request to send */
#define		TIOCM_ST	0010		/* secondary transmit */
#define		TIOCM_SR	0020		/* secondary receive */
#define		TIOCM_CTS	0040		/* clear to send */
#define		TIOCM_CAR	0100		/* carrier detect */
#define		TIOCM_CD	TIOCM_CAR
#define		TIOCM_RNG	0200		/* ring */
#define		TIOCM_RI	TIOCM_RNG
#define		TIOCM_DSR	0400		/* data set ready */
						/* 8-10 compat */
#define	TIOCEXCL	_IO('t', 13)		/* set exclusive use of tty */
#define	TIOCNXCL	_IO('t', 14)		/* reset exclusive use of tty */
						/* 15 unused */
#define	TIOCFLUSH	_IOW('t', 16, int)	/* flush buffers */
						/* 17-18 compat */
#define	TIOCGETA	_IOR('t', 19, struct termios) /* get termios struct */
#define	TIOCSETA	_IOW('t', 20, struct termios) /* set termios struct */
#define	TIOCSETAW	_IOW('t', 21, struct termios) /* drain output, set */
#define	TIOCSETAF	_IOW('t', 22, struct termios) /* drn out, fls in, set */
#define	TIOCGETD	_IOR('t', 26, int)	/* get line discipline */
#define	TIOCSETD	_IOW('t', 27, int)	/* set line discipline */
						/* 127-124 compat */
#define	TIOCSBRK	_IO('t', 123)		/* set break bit */
#define	TIOCCBRK	_IO('t', 122)		/* clear break bit */
#define	TIOCSDTR	_IO('t', 121)		/* set data terminal ready */
#define	TIOCCDTR	_IO('t', 120)		/* clear data terminal ready */
#define	TIOCGPGRP	_IOR('t', 119, int)	/* get pgrp of tty */
#define	TIOCSPGRP	_IOW('t', 118, int)	/* set pgrp of tty */
						/* 117-116 compat */
#define	TIOCOUTQ	_IOR('t', 115, int)	/* output queue size */
#define	TIOCSTI		_IOW('t', 114, char)	/* simulate terminal input */
#define	TIOCNOTTY	_IO('t', 113)		/* void tty association */
#define	TIOCPKT		_IOW('t', 112, int)	/* pty: set/clear packet mode */
#define		TIOCPKT_DATA		0x00	/* data packet */
#define		TIOCPKT_FLUSHREAD	0x01	/* flush packet */
#define		TIOCPKT_FLUSHWRITE	0x02	/* flush packet */
#define		TIOCPKT_STOP		0x04	/* stop output */
#define		TIOCPKT_START		0x08	/* start output */
#define		TIOCPKT_NOSTOP		0x10	/* no more ^S, ^Q */
#define		TIOCPKT_DOSTOP		0x20	/* now do ^S ^Q */
#define		TIOCPKT_IOCTL		0x40	/* state change of pty driver */
#define	TIOCSTOP	_IO('t', 111)		/* stop output, like ^S */
#define	TIOCSTART	_IO('t', 110)		/* start output, like ^Q */
#define	TIOCMSET	_IOW('t', 109, int)	/* set all modem bits */
#define	TIOCMBIS	_IOW('t', 108, int)	/* bis modem bits */
#define	TIOCMBIC	_IOW('t', 107, int)	/* bic modem bits */
#define	TIOCMGET	_IOR('t', 106, int)	/* get all modem bits */
#define	TIOCREMOTE	_IOW('t', 105, int)	/* remote input editing */
#define	TIOCGWINSZ	_IOR('t', 104, struct winsize)	/* get window size */
#define	TIOCSWINSZ	_IOW('t', 103, struct winsize)	/* set window size */
#define	TIOCUCNTL	_IOW('t', 102, int)	/* pty: set/clr usr cntl mode */
#define		UIOCCMD(n)	_IO('u', n)		/* usr cntl op "n" */
#define	TIOCCONS	_IOW('t', 98, int)		/* become virtual console */
#define	TIOCSCTTY	_IO('t', 97)		/* become controlling tty */
#define	TIOCEXT		_IOW('t', 96, int)	/* pty: external processing */
#define	TIOCSIG		_IO('t', 95)		/* pty: generate signal */
#define TIOCDRAIN	_IO('t', 94)		/* wait till output drained */

#define TTYDISC		0		/* termios tty line discipline */
#define	TABLDISC	3		/* tablet discipline */
#define	SLIPDISC	4		/* serial IP discipline */


#define	FIOCLEX		_IO('f', 1)		/* set close on exec on fd */
#define	FIONCLEX	_IO('f', 2)		/* remove close on exec */
#define	FIONREAD	_IOR('f', 127, int)	/* get # bytes to read */
#define	FIONBIO		_IOW('f', 126, int)	/* set/clear non-blocking i/o */
#define	FIOASYNC	_IOW('f', 125, int)	/* set/clear async i/o */
#define	FIOSETOWN	_IOW('f', 124, int)	/* set owner */
#define	FIOGETOWN	_IOR('f', 123, int)	/* get owner */

/* socket i/o controls */
#define	SIOCSHIWAT	_IOW('s',  0, int)		/* set high watermark */
#define	SIOCGHIWAT	_IOR('s',  1, int)		/* get high watermark */
#define	SIOCSLOWAT	_IOW('s',  2, int)		/* set low watermark */
#define	SIOCGLOWAT	_IOR('s',  3, int)		/* get low watermark */
#define	SIOCATMARK	_IOR('s',  7, int)		/* at oob mark? */
#define	SIOCSPGRP	_IOW('s',  8, int)		/* set process group */
#define	SIOCGPGRP	_IOR('s',  9, int)		/* get process group */

#define	SIOCADDRT	_IOW('r', 10, struct ortentry)	/* add route */
#define	SIOCDELRT	_IOW('r', 11, struct ortentry)	/* delete route */

#define	SIOCSIFADDR	_IOW('i', 12, struct ifreq)	/* set ifnet address */
#define	OSIOCGIFADDR	_IOWR('i',13, struct ifreq)	/* get ifnet address */
#define	SIOCGIFADDR	_IOWR('i',33, struct ifreq)	/* get ifnet address */
#define	SIOCGIFHWADDR	_IOWR('i',39, struct ifreq)	/* get hwaddress */
#define	SIOCSIFDSTADDR	_IOW('i', 14, struct ifreq)	/* set p-p address */
#define	OSIOCGIFDSTADDR	_IOWR('i',15, struct ifreq)	/* get p-p address */
#define	SIOCGIFDSTADDR	_IOWR('i',34, struct ifreq)	/* get p-p address */
#define	SIOCSIFFLAGS	_IOW('i', 16, struct ifreq_short)/* set ifnet flags */
#define	SIOCGIFFLAGS	_IOWR('i',17, struct ifreq_short)/* get ifnet flags */
#define	OSIOCGIFBRDADDR	_IOWR('i',18, struct ifreq)	/* get broadcast addr */
#define	SIOCGIFBRDADDR	_IOWR('i',35, struct ifreq)	/* get broadcast addr */
#define	SIOCSIFBRDADDR	_IOW('i',19, struct ifreq)	/* set broadcast addr */
#define	OSIOCGIFCONF	_IOWR('i',20, struct ifconf)	/* get ifnet list */
#define	SIOCGIFCONF	_IOWR('i',36, struct ifconf)	/* get ifnet list */
#define	OSIOCGIFNETMASK	_IOWR('i',21, struct ifreq)	/* get net addr mask */
#define	SIOCGIFNETMASK	_IOWR('i',37, struct ifreq)	/* get net addr mask */
#define	SIOCSIFNETMASK	_IOW('i',22, struct ifreq)	/* set net addr mask */
#define	SIOCGIFMETRIC	_IOWR('i',23, struct ifreq_int)	/* get IF metric */
#define	SIOCSIFMETRIC	_IOW('i',24, struct ifreq_int)	/* set IF metric */
#define	SIOCDIFADDR	_IOW('i',25, struct ifreq)	/* delete IF addr */
#define	SIOCAIFADDR	_IOW('i',26, struct ifaliasreq)	/* add/chg IF alias */

#define	SIOCSARP	_IOW('i', 30, struct arpreq)	/* set arp entry */
#define	OSIOCGARP	_IOWR('i',31, struct arpreq)	/* get arp entry */
#define	SIOCGARP	_IOWR('i',38, struct arpreq)	/* get arp entry */
#define	SIOCDARP	_IOW('i', 32, struct arpreq)	/* delete arp entry */

#define SIOCGIFMTU	_IOWR('i', 51, struct ifreq_int)/* get IF mtu */
#define SIOCSIFMTU	_IOW('i', 52, struct ifreq_int)	/* set IF mtu */

#define SIOCGIFINDEX	_IOWR('i', 90, struct ifreq_int)/* get IF index */
#define SIOCGIFNAME	_IOWR('i', 91, struct ifreq_int)/* set IF name */


/* Compatibility with 4.3 BSD terminal driver.
   From 4.4 <sys/ioctl_compat.h>.  */

#ifdef __USE_MISC
#ifdef USE_OLD_TTY
# undef  TIOCGETD
# define TIOCGETD	_IOR('t', 0, int)	/* get line discipline */
# undef  TIOCSETD
# define TIOCSETD	_IOW('t', 1, int)	/* set line discipline */
#else
# define OTIOCGETD	_IOR('t', 0, int)	/* get line discipline */
# define OTIOCSETD	_IOW('t', 1, int)	/* set line discipline */
#endif
#define	TIOCHPCL	_IO('t', 2)		/* hang up on last close */
#define	TIOCGETP	_IOR('t', 8,struct sgttyb)/* get parameters -- gtty */
#define	TIOCSETP	_IOW('t', 9,struct sgttyb)/* set parameters -- stty */
#define	TIOCSETN	_IOW('t',10,struct sgttyb)/* as above, but no flushtty*/
#define	TIOCSETC	_IOW('t',17,struct tchars)/* set special characters */
#define	TIOCGETC	_IOR('t',18,struct tchars)/* get special characters */
#define		TANDEM		0x00000001	/* send stopc on out q full */
#define		CBREAK		0x00000002	/* half-cooked mode */
#define		LCASE		0x00000004	/* simulate lower case */
#define		ECHO		0x00000008	/* echo input */
#define		CRMOD		0x00000010	/* map \r to \r\n on output */
#define		RAW		0x00000020	/* no i/o processing */
#define		ODDP		0x00000040	/* get/send odd parity */
#define		EVENP		0x00000080	/* get/send even parity */
#define		ANYP		0x000000c0	/* get any parity/send none */
#define		NLDELAY		0x00000300	/* \n delay */
#define			NL0	0x00000000
#define			NL1	0x00000100	/* tty 37 */
#define			NL2	0x00000200	/* vt05 */
#define			NL3	0x00000300
#define		TBDELAY		0x00000c00	/* horizontal tab delay */
#define			TAB0	0x00000000
#define			TAB1	0x00000400	/* tty 37 */
#define			TAB2	0x00000800
#define		XTABS		0x00000c00	/* expand tabs on output */
#define		CRDELAY		0x00003000	/* \r delay */
#define			CR0	0x00000000
#define			CR1	0x00001000	/* tn 300 */
#define			CR2	0x00002000	/* tty 37 */
#define			CR3	0x00003000	/* concept 100 */
#define		VTDELAY		0x00004000	/* vertical tab delay */
#define			FF0	0x00000000
#define			FF1	0x00004000	/* tty 37 */
#define		BSDELAY		0x00008000	/* \b delay */
#define			BS0	0x00000000
#define			BS1	0x00008000
#define		ALLDELAY	(NLDELAY|TBDELAY|CRDELAY|VTDELAY|BSDELAY)
#define		CRTBS		0x00010000	/* do backspacing for crt */
#define		PRTERA		0x00020000	/* \ ... / erase */
#define		CRTERA		0x00040000	/* " \b " to wipe out char */
#define		TILDE		0x00080000	/* hazeltine tilde kludge */
#define		MDMBUF		0x00100000	/*start/stop output on carrier*/
#define		LITOUT		0x00200000	/* literal output */
#define		TOSTOP		0x00400000	/*SIGSTOP on background output*/
#define		FLUSHO		0x00800000	/* flush output to terminal */
#define		NOHANG		0x01000000	/* (no-op) was no SIGHUP on carrier drop */
#define		L001000		0x02000000
#define		CRTKIL		0x04000000	/* kill line with " \b " */
#define		PASS8		0x08000000
#define		CTLECH		0x10000000	/* echo control chars as ^X */
#define		PENDIN		0x20000000	/* tp->t_rawq needs reread */
#define		DECCTQ		0x40000000	/* only ^Q starts after ^S */
#define		NOFLSH		0x80000000	/* no output flush on signal */
#define	TIOCLBIS	_IOW('t', 127, int)	/* bis local mode bits */
#define	TIOCLBIC	_IOW('t', 126, int)	/* bic local mode bits */
#define	TIOCLSET	_IOW('t', 125, int)	/* set entire local mode word */
#define	TIOCLGET	_IOR('t', 124, int)	/* get local modes */
#define		LCRTBS		(CRTBS>>16)
#define		LPRTERA		(PRTERA>>16)
#define		LCRTERA		(CRTERA>>16)
#define		LTILDE		(TILDE>>16)
#define		LMDMBUF		(MDMBUF>>16)
#define		LLITOUT		(LITOUT>>16)
#define		LTOSTOP		(TOSTOP>>16)
#define		LFLUSHO		(FLUSHO>>16)
#define		LNOHANG		(NOHANG>>16)
#define		LCRTKIL		(CRTKIL>>16)
#define		LPASS8		(PASS8>>16)
#define		LCTLECH		(CTLECH>>16)
#define		LPENDIN		(PENDIN>>16)
#define		LDECCTQ		(DECCTQ>>16)
#define		LNOFLSH		(NOFLSH>>16)
#define	TIOCSLTC	_IOW('t',117,struct ltchars)/* set local special chars*/
#define	TIOCGLTC	_IOR('t',116,struct ltchars)/* get local special chars*/
#define OTIOCCONS	_IO('t', 98)	/* for hp300 -- sans int arg */
#define	OTTYDISC	0
#define	NETLDISC	1
#define	NTTYDISC	2

/* From 4.4 <sys/ttydev.h>.   */
#ifdef USE_OLD_TTY
# define B0	0
# define B50	1
# define B75	2
# define B110	3
# define B134	4
# define B150	5
# define B200	6
# define B300	7
# define B600	8
# define B1200	9
# define B1800	10
# define B2400	11
# define B4800	12
# define B9600	13
# define EXTA	14
# define EXTB	15
#endif /* USE_OLD_TTY */
#endif

#endif /* bits/ioctls.h */
