#include <sysdep.h>

	.text
ENTRY(bcopy)
	xchg	%rdi, %rsi
	jmp	__libc_memmove	/* Branch to IFUNC memmove.  */
END(bcopy)
