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
 * CARNEGIE MELLON ALLOWS FREE USE OF THIS SOFTWARE IN ITS
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
 * any improvements or extensions that they make and grant Carnegie the
 * rights to redistribute these changes.
 */
/*
 * (pre-GNU) HISTORY
 *
 * Revision 2.3  92/03/31  15:18:52  rpd
 * 	Added KERN_DEVICE_MOD for device errors.
 * 	[92/03/09            rpd]
 *
 * Revision 2.2  92/01/16  00:21:17  rpd
 * 	Moved from user collection to mk collection.
 *
 * Revision 2.2  91/03/27  15:37:37  mrt
 * 	First checkin
 *
 */
/*
 *	File:	errorlib.h
 *	Author:	Douglas Orr, Carnegie Mellon University
 *	Date:	Mar. 1988
 *
 *	Error bases for subsytems errors.
 */

#include <mach/error.h>

#define KERN_DEVICE_MOD		(err_kern|err_sub(1))

#define BOOTSTRAP_FS_MOD	(err_bootstrap|err_sub(0))

#define	MACH_IPC_SEND_MOD	(err_mach_ipc|err_sub(0))
#define	MACH_IPC_RCV_MOD	(err_mach_ipc|err_sub(1))
#define	MACH_IPC_MIG_MOD	(err_mach_ipc|err_sub(2))

#define	IPC_SEND_MOD		(err_ipc|err_sub(0))
#define	IPC_RCV_MOD		(err_ipc|err_sub(1))
#define	IPC_MIG_MOD		(err_ipc|err_sub(2))

#define	SERV_NETNAME_MOD	(err_server|err_sub(0))
#define	SERV_ENV_MOD		(err_server|err_sub(1))
#define	SERV_EXECD_MOD		(err_server|err_sub(2))


#define	NO_SUCH_ERROR		"unknown error code"

struct error_subsystem {
	const char		* subsys_name;
	int			max_code;
	const char		* const * codes;
};

struct error_system {
	int			max_sub;
	const char		* bad_sub;
	const struct error_subsystem	* subsystem;
};

#define errors __mach_error_systems
extern const struct error_system 	errors[err_max_system+1];

#define	errlib_count(s)		(sizeof(s)/sizeof(s[0]))
