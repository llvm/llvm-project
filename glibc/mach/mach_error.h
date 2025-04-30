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
 * Revision 2.2  92/01/16  00:08:10  rpd
 * 	Moved from user collection to mk collection.
 *
 * Revision 2.2  91/03/27  15:39:13  mrt
 * 	First checkin
 *
 */
/*
 *	File:	mach_error.h
 *	Author:	Douglas Orr, Carnegie Mellon University
 *	Date:	Mar. 1988
 *
 *	Definitions of routines in mach_error.c
 */

#ifndef	_MACH_ERROR_
#define	_MACH_ERROR_	1

#include <mach/error.h>

const char	*mach_error_string(
/*
 *	Returns a string appropriate to the error argument given
 */
	mach_error_t error_value
				);

void		mach_error(
/*
 *	Prints an appropriate message on the standard error stream
 */
	char 		*str,
	mach_error_t	error_value
				);

const char	*mach_error_type(
/*
 *	Returns a string with the error system, subsystem and code
*/
	mach_error_t	error_value
				);

#endif	/* _MACH_ERROR_ */
