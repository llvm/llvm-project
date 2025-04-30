/*
 * Sparc v8 has divide.
 */

#include <sysdep.h>

ENTRY(.div)

	sra	%o0, 31, %o2
	wr	%o2, 0, %y
	nop
	nop
	nop
	sdivcc	%o0, %o1, %o0
	bvs,a	1f
	 xnor	%o0, %g0, %o0
1:	retl
	 nop

END(.div)
