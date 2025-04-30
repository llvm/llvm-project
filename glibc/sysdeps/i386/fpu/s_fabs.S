#include <sysdep.h>
#include <libm-alias-double.h>

	.text
ENTRY(__fabs)
	fldl	4(%esp)
	fabs
	ret
END(__fabs)
libm_alias_double (__fabs, fabs)
