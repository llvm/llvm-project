/*
 * Sparc v8 has divide.
 */

#include <sysdep.h>

ENTRY(.rem)

	sra	%o0, 31, %o2
	wr	%o2, 0, %y
	nop
	nop
	nop
	sdivcc	%o0, %o1, %o2
	bvs,a	1f
	 xnor	%o2, %g0, %o2
1:	smul	%o2, %o1, %o2
	retl
	 sub	%o0, %o2, %o0

END(.rem)
