#include <sysdep.h>
#include <libm-alias-float.h>

	.text
ENTRY(__fabsf)
	flds	4(%esp)
	fabs
	ret
END(__fabsf)
libm_alias_float (__fabs, fabs)
