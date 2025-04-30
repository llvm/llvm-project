/*
 * Sparc v9 has divide.
 * As divx takes 68 cycles and udiv only 37,
 * we use udiv eventhough it is deprecated.
 */

#include <sysdep.h>

	.text
	.align		32
ENTRY(.urem)

	wr		%g0, 0, %y
	udiv		%o0, %o1, %o2
	umul		%o2, %o1, %o2
	retl
	 sub		%o0, %o2, %o0

END(.urem)
