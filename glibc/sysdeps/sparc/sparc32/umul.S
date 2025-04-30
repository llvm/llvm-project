/*
 * Sparc v8 has multiply.
 */

#include <sysdep.h>

ENTRY(.umul)

	umul	%o0, %o1, %o0
	retl
	 rd	%y, %o1

END(.umul)
