/*
 * Mach Operating System
 * Copyright (c) 1991,1990,1989 Carnegie Mellon University
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

/* This file was broken out from:

	Revision 2.3  92/04/01  19:38:18  rpd

   The static do_compat function is renamed to be globally accessible.
 */

#include <mach/error.h>
#include <mach_error.h>
#include <errorlib.h>


void
__mach_error_map_compat(mach_error_t  *org_err)
{
	mach_error_t		err = *org_err;

	/*
	 * map old error numbers to
	 * to new error sys & subsystem
	 */

	if ((-200 < err) && (err <= -100))
		err = -(err + 100) | IPC_SEND_MOD;
	else if ((-300 < err) && (err <= -200))
		err = -(err + 200) | IPC_RCV_MOD;
	else if ((-400 < err) && (err <= -300))
		err = -(err + 300) | MACH_IPC_MIG_MOD;
	else if ((1000 <= err) && (err < 1100))
		err = (err - 1000) | SERV_NETNAME_MOD;
	else if ((1600 <= err) && (err < 1700))
		err = (err - 1600) | SERV_ENV_MOD;
	else if ((27600 <= err) && (err < 27700))
		err = (err - 27600) | SERV_EXECD_MOD;
	else if ((2500 <= err) && (err < 2600))
		err = (err - 2500) | KERN_DEVICE_MOD;
	else if ((5000 <= err) && (err < 5100))
		err = (err - 5000) | BOOTSTRAP_FS_MOD;

	*org_err = err;
}
