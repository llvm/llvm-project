/*
 * Sparc v8 has divide.
 */

#include <sysdep.h>

ENTRY(.udiv)

	wr	%g0, 0, %y
	nop
	nop
	retl
	 udiv	%o0, %o1, %o0

END(.udiv)
strong_alias (.udiv, __wrap_.udiv)
