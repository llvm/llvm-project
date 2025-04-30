/*
 * Sparc v9 has divide.
 * As divx takes 68 cycles and udiv only 37,
 * we use udiv eventhough it is deprecated.
 */

#include <sysdep.h>

	.text
	.align		32
ENTRY(.udiv)

	wr		%g0, 0, %y
	retl
	 udiv		%o0, %o1, %o0

END(.udiv)
strong_alias (.udiv, __wrap_.udiv)
