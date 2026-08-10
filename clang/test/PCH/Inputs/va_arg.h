#include <stdarg.h>

typedef __builtin_ms_va_list __ms_va_list;
#define __ms_va_start(ap, a) __builtin_ms_va_start(ap, a)
#define __ms_va_end(ap) __builtin_ms_va_end(ap)
