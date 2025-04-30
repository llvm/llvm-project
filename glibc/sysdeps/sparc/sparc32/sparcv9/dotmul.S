/*
 * Sparc v9 has multiply.
 */

#include <sysdep.h>

	.text
	.align		32
ENTRY(.mul)

	sra		%o0, 0, %o0
	sra		%o1, 0, %o1
	mulx		%o0, %o1, %o0
	retl
	 srax		%o0, 32, %o1

END(.mul)
