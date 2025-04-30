#include <libm-alias-ldouble.h>
#include <sysdep.h>

	.text
ENTRY(__fabsl)
	fldt	4(%esp)
	fabs
	ret
END(__fabsl)
libm_alias_ldouble (__fabs, fabs)
