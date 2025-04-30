/*
 * Mach Operating System
 * Copyright (C) 1993,1991,1990 Carnegie Mellon University
 * All Rights Reserved.
 *
 * Permission to use, copy, modify and distribute this software and its
 * documentation is hereby granted, provided that both the copyright
 * notice and this permission notice appear in all copies of the
 * software, derivative works or modified versions, and any portions
 * thereof, and that both notices appear in supporting documentation.
 *
 * CARNEGIE MELLON ALLOWS FREE USE OF THIS SOFTWARE IN ITS "AS IS"
 * CONDITION.  CARNEGIE MELLON DISCLAIMS ANY LIABILITY OF ANY KIND FOR
 * ANY DAMAGES WHATSOEVER RESULTING FROM THE USE OF THIS SOFTWARE.
 *
 * Carnegie Mellon requests users of this software to return to
 *
 *  Software Distribution Coordinator  or  Software.Distribution@CS.CMU.EDU
 *  School of Computer Science
 *  Carnegie Mellon University
 *  Pittsburgh PA 15213-3890
 *
 * any improvements or extensions that they make and grant Carnegie Mellon
 * the rights to redistribute these changes.
 */
/*
 * (pre-GNU) HISTORY
 *
 * Revision 2.8  93/03/11  13:46:40  danner
 * 	unsigned long -> unsigned int.
 * 	[93/03/09            danner]
 *
 * Revision 2.7  92/05/21  17:25:11  jfriedl
 * 	Appended 'U' to constants that would otherwise be signed.
 * 	[92/05/16            jfriedl]
 *
 * Revision 2.6  91/06/19  11:59:44  rvb
 * 	Second byte of boothowto is flags for "startup" program.
 * 	[91/06/18            rvb]
 * 	Add ifndef __ASSEMBLER__ so that vax_init.s can include it.
 * 	[91/06/11            rvb]
 *
 * Revision 2.5  91/05/14  17:40:11  mrt
 * 	Correcting copyright
 *
 * Revision 2.4  91/02/05  17:56:48  mrt
 * 	Changed to new Mach copyright
 * 	[91/02/01  17:49:12  mrt]
 *
 * Revision 2.3  90/08/27  22:12:56  dbg
 * 	Added definitions used by Mach Kernel: RB_DEBUGGER, RB_UNIPROC,
 * 	RB_NOBOOTRC, RB_ALTBOOT.  Moved RB_KDB to 0x04 (Mach value).
 * 	Removed RB_RDONLY, RB_DUMP, RB_NOSYNC.
 * 	[90/08/14            dbg]
 *
 */

/*
   Copyright (C) 1982, 1986, 1988 Regents of the University of California.
   All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   1. Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
   2. Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
   4. Neither the name of the University nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
   ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
   ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
   FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
   OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
   HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
   LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
   OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
   SUCH DAMAGE.*/

/*
 *	@(#)reboot.h	7.5 (Berkeley) 6/27/88
 */

#ifndef	_SYS_REBOOT_H_
#define	_SYS_REBOOT_H_

#include <features.h>

/*
 * Arguments to reboot system call.
 * These are converted to switches, and passed to startup program,
 * and on to init.
 */
#define	RB_AUTOBOOT	0	/* flags for system auto-booting itself */

#define	RB_ASKNAME	0x01	/* -a: ask for file name to reboot from */
#define	RB_SINGLE	0x02	/* -s: reboot to single user only */
#define	RB_KDB		0x04	/* -d: kernel debugger symbols loaded */
#define	RB_HALT		0x08	/* -h: enter KDB at bootup */
				/*     for host_reboot(): don't reboot,
				       just halt */
#define	RB_INITNAME	0x10	/* -i: name given for /etc/init (unused) */
#define	RB_DFLTROOT	0x20	/*     use compiled-in rootdev */
#define	RB_NOBOOTRC	0x20	/* -b: don't run /etc/rc.boot */
#define RB_ALTBOOT	0x40	/*     use /boot.old vs /boot */
#define	RB_UNIPROC	0x80	/* -u: start only one processor */

#define	RB_SHIFT	8	/* second byte is for ux */

#define	RB_DEBUGGER	0x1000	/*     for host_reboot(): enter kernel
				       debugger from user level */

/*
 * Constants for converting boot-style device number to type,
 * adaptor (uba, mba, etc), unit number and partition number.
 * Type (== major device number) is in the low byte
 * for backward compatibility.  Except for that of the "magic
 * number", each mask applies to the shifted value.
 * Format:
 *	 (4) (4) (4) (4)  (8)     (8)
 *	--------------------------------
 *	|MA | AD| CT| UN| PART  | TYPE |
 *	--------------------------------
 */
#define	B_ADAPTORSHIFT		24
#define	B_ADAPTORMASK		0x0f
#define	B_ADAPTOR(val)		(((val) >> B_ADAPTORSHIFT) & B_ADAPTORMASK)
#define B_CONTROLLERSHIFT	20
#define B_CONTROLLERMASK	0xf
#define	B_CONTROLLER(val)	(((val)>>B_CONTROLLERSHIFT) & B_CONTROLLERMASK)
#define B_UNITSHIFT		16
#define B_UNITMASK		0xf
#define	B_UNIT(val)		(((val) >> B_UNITSHIFT) & B_UNITMASK)
#define B_PARTITIONSHIFT	8
#define B_PARTITIONMASK		0xff
#define	B_PARTITION(val)	(((val) >> B_PARTITIONSHIFT) & B_PARTITIONMASK)
#define	B_TYPESHIFT		0
#define	B_TYPEMASK		0xff
#define	B_TYPE(val)		(((val) >> B_TYPESHIFT) & B_TYPEMASK)

#define	B_MAGICMASK	0xf0000000U
#define	B_DEVMAGIC	0xa0000000U

#define MAKEBOOTDEV(type, adaptor, controller, unit, partition) \
	(((type) << B_TYPESHIFT) | ((adaptor) << B_ADAPTORSHIFT) | \
	((controller) << B_CONTROLLERSHIFT) | ((unit) << B_UNITSHIFT) | \
	((partition) << B_PARTITIONSHIFT) | B_DEVMAGIC)


#ifdef	KERNEL
#ifndef	__ASSEMBLER__
extern int boothowto;
#endif	/* __ASSEMBLER__ */
#endif

__BEGIN_DECLS

/* Reboot or halt the system.  */
extern int reboot (int __howto) __THROW;

__END_DECLS


#endif	/* _SYS_REBOOT_H_ */
