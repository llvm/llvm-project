/*
 * Sparc v9 has multiply.
 */

#include <sysdep.h>

	.text
	.align		32
ENTRY(.umul)

	srl		%o0, 0, %o0
	srl		%o1, 0, %o1
	mulx		%o0, %o1, %o0
	retl
	 srlx		%o0, 32, %o1

END(.umul)
