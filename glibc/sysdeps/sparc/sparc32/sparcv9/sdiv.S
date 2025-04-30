/*
 * Sparc v9 has divide.
 * As divx takes 68 cycles and sdivcc only 36,
 * we use sdivcc eventhough it is deprecated.
 */

#include <sysdep.h>

	.text
	.align		32
ENTRY(.div)

	sra		%o0, 31, %o2
	wr		%o2, 0, %y
	sdivcc		%o0, %o1, %o0
	xnor		%o0, %g0, %o2
	retl
	 movvs		%icc, %o2, %o0

END(.div)
