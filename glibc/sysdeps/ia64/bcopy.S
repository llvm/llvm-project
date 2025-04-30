#include <sysdep.h>

ENTRY(bcopy)
	.regstk 3, 0, 0, 0
	mov r8 = in0
	mov in0 = in1
	;;
	mov in1 = r8
	br.cond.sptk.many HIDDEN_BUILTIN_JUMPTARGET(memmove)
END(bcopy)
