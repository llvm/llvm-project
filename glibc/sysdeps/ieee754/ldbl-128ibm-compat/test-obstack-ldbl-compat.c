#define _FORTIFY_SOURCE 0
#define OBSTACK_FUNCTION obstack_printf
#define OBSTACK_FUNCTION_PARAMS (&ob, "%.30Lf", ld)
#define VOBSTACK_FUNCTION obstack_vprintf
#define VOBSTACK_FUNCTION_PARAMS (&ob, "%.30Lf", ap)
#include <test-obstack-ldbl-compat-template.c>
