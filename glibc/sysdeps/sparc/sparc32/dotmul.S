/*
 * Sparc v8 has multiply.
 */

#include <sysdep.h>

ENTRY(.mul)

	smul	%o0, %o1, %o0
	retl
	 rd	%y, %o1

END(.mul)
