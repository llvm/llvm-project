/*
 * Sparc v9 has divide.
 * As divx takes 68 cycles and sdivcc only 36,
 * we use sdivcc eventhough it is deprecated.
 */

#include <sysdep.h>

	.text
	.align		32
ENTRY(.rem)

	sra		%o0, 31, %o2
	wr		%o2, 0, %y
	sdivcc		%o0, %o1, %o2
	xnor		%o2, %g0, %o3
	movvs		%icc, %o3, %o2
	smul		%o2, %o1, %o2
	retl
	 sub		%o0, %o2, %o0

END(.rem)
