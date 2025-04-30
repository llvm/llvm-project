#ifndef _ERROR_H
#include <misc/error.h>

#include <stdarg.h>

void
__error_internal (int status, int errnum, const char *message,
		  va_list args, unsigned int mode_flags);

void
__error_at_line_internal (int status, int errnum, const char *file_name,
			  unsigned int line_number, const char *message,
			  va_list args, unsigned int mode_flags);

#endif
