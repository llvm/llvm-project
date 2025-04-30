/* Copy memory to memory until the specified number of bytes
   has been copied.  Overlap is handled correctly.
   For SPARC V9.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <sysdep.h>

#ifndef XCC
# define XCC    xcc
	.register	%g2, #scratch
#endif

ENTRY(memmove)
	mov	%o0, %g2	/* Save pointer to destination  */
	cmp	%o1, %o0	/* if from address is >= to use forward copy  */
	bgeu,a	%XCC, 2f	/* else use backward if ...  */
	 cmp	%o2, 17		/* delay slot, for small counts copy bytes  */

	sub	%o0, %o1, %o4	/* get difference of two addresses  */
	cmp	%o2, %o4	/* compare size and difference of addresses  */
	bgu	%XCC, .Lovbc	/* if size is bigger, have to do overlapped copy  */
	 cmp	%o2, 17		/* delay slot, for small counts copy bytes  */
/*
 * normal, copy forwards
 */
2:	ble	%XCC, .Ldbytecp
	 andcc	%o1, 3, %o5	/* is src word aligned  */
	bz,pn	%icc, .Laldst
	 cmp	%o5, 2		/* is src half-word aligned  */
	be,pn	%icc, .Ls2alg
	 cmp	%o5, 3		/* src is byte aligned  */
	ldub	[%o1], %o3	/* move 1 or 3 bytes to align it  */
	inc	1, %o1
	stb	%o3, [%o0]	/* move a byte to align src  */
	inc	1, %o0
	bne,pn	%icc, .Ls2alg
	 dec	%o2
	b	.Lald		/* now go align dest  */
	 andcc	%o0, 3, %o5

.Ls2alg:
	lduh	[%o1], %o3	/* know src is 2 byte aligned  */
	inc	2, %o1
	srl	%o3, 8, %o4
	stb	%o4, [%o0]	/* have to do bytes,  */
	stb	%o3, [%o0 + 1]	/* don't know dst alingment  */
	inc	2, %o0
	dec	2, %o2

.Laldst:
	andcc	%o0, 3, %o5	/* align the destination address  */
.Lald:	bz,pn	%icc, .Lw4cp
	 cmp	%o5, 2
	bz,pn	%icc, .Lw2cp
	 cmp	%o5, 3
.Lw3cp:
	lduw	[%o1], %o4
	inc	4, %o1
	srl	%o4, 24, %o5
	stb	%o5, [%o0]
	bne,pt	%icc, .Lw1cp
	 inc	%o0
	dec	1, %o2
	andn	%o2, 3, %o3	/* i3 is aligned word count  */
	dec	4, %o3		/* avoid reading beyond tail of src  */
	sub	%o1, %o0, %o1	/* i1 gets the difference  */

1:	sll	%o4, 8, %g1	/* save residual bytes  */
	lduw	[%o1+%o0], %o4
	deccc	4, %o3
	srl	%o4, 24, %o5	/* merge with residual  */
	or	%o5, %g1, %g1
	st	%g1, [%o0]
	bnz,pt	%XCC, 1b
	 inc	4, %o0
	sub	%o1, 3, %o1	/* used one byte of last word read  */
	and	%o2, 3, %o2
	b	7f
	 inc	4, %o2

.Lw1cp:
	srl	%o4, 8, %o5
	sth	%o5, [%o0]
	inc	2, %o0
	dec	3, %o2
	andn	%o2, 3, %o3
	dec	4, %o3		/* avoid reading beyond tail of src  */
	sub	%o1, %o0, %o1	/* i1 gets the difference  */

2:	sll	%o4, 24, %g1	/* save residual bytes  */
	lduw	[%o1+%o0], %o4
	deccc	4, %o3
	srl	%o4, 8, %o5	/* merge with residual  */
	or	%o5, %g1, %g1
	st	%g1, [%o0]
	bnz,pt	%XCC, 2b
	 inc	4, %o0
	sub	%o1, 1, %o1	/* used three bytes of last word read  */
	and	%o2, 3, %o2
	b	7f
	inc	4, %o2

.Lw2cp:
	lduw	[%o1], %o4
	inc	4, %o1
	srl	%o4, 16, %o5
	sth	%o5, [%o0]
	inc	2, %o0
	dec	2, %o2
	andn	%o2, 3, %o3	/* i3 is aligned word count  */
	dec	4, %o3		/* avoid reading beyond tail of src  */
	sub	%o1, %o0, %o1	/* i1 gets the difference  */

3:	sll	%o4, 16, %g1	/* save residual bytes  */
	lduw	[%o1+%o0], %o4
	deccc	4, %o3
	srl	%o4, 16, %o5	/* merge with residual  */
	or	%o5, %g1, %g1
	st	%g1, [%o0]
	bnz,pt	%XCC, 3b
	 inc	4, %o0
	sub	%o1, 2, %o1	/* used two bytes of last word read  */
	and	%o2, 3, %o2
	b	7f
	 inc	4, %o2

.Lw4cp:
	andn	%o2, 3, %o3	/* i3 is aligned word count  */
	sub	%o1, %o0, %o1	/* i1 gets the difference  */

1:	lduw	[%o1+%o0], %o4	/* read from address  */
	deccc	4, %o3		/* decrement count  */
	st	%o4, [%o0]	/* write at destination address  */
	bg,pt	%XCC, 1b
	 inc	4, %o0		/* increment to address  */
	b	7f
	 and	%o2, 3, %o2	/* number of leftover bytes, if any  */

/*
 * differenced byte copy, works with any alignment
 */
.Ldbytecp:
	b	7f
	 sub	%o1, %o0, %o1	/* i1 gets the difference  */

4:	stb	%o4, [%o0]	/* write to address  */
	inc	%o0		/* inc to address  */
7:	deccc	%o2		/* decrement count  */
	bge,a	%XCC, 4b	/* loop till done  */
	 ldub	[%o1+%o0], %o4	/* read from address  */
	retl
	 mov	%g2, %o0	/* return pointer to destination  */

/*
 * an overlapped copy that must be done "backwards"
 */
.Lovbc:
	add	%o1, %o2, %o1	/* get to end of source space  */
	add	%o0, %o2, %o0	/* get to end of destination space  */
	sub	%o1, %o0, %o1	/* i1 gets the difference  */

5:	dec	%o0		/* decrement to address  */
	ldub	[%o1+%o0], %o3	/* read a byte  */
	deccc	%o2		/* decrement count  */
	bg,pt	%XCC, 5b 	/* loop until done  */
	 stb	%o3, [%o0]	/* write byte  */
	retl
	 mov	%g2, %o0	/* return pointer to destination  */
END(memmove)

libc_hidden_builtin_def (memmove)
