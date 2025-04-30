#define _FORTIFY_SOURCE 2
#define OBSTACK_FUNCTION __obstack_printf_chk
#define OBSTACK_FUNCTION_PARAMS (&ob, 1, "%.30Lf", ld)
#define VOBSTACK_FUNCTION __obstack_vprintf_chk
#define VOBSTACK_FUNCTION_PARAMS (&ob, 1, "%.30Lf", ap)
#include <test-obstack-ldbl-compat-template.c>
