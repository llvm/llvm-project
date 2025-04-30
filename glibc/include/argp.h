#ifndef _ARGP_H
#include <argp/argp.h>

/* Prototypes for internal argp.h functions.  */
#include <stdarg.h>
void
__argp_error_internal (const struct argp_state *state, const char *fmt,
		       va_list ap, unsigned int mode_flags);

void
__argp_failure_internal (const struct argp_state *state, int status,
			 int errnum, const char *fmt, va_list ap,
			 unsigned int mode_flags);

#ifndef _ISOMAC
extern __typeof (__argp_error) __argp_error attribute_hidden;
extern __typeof (__argp_failure) __argp_failure attribute_hidden;
extern __typeof (__argp_input) __argp_input attribute_hidden;
extern __typeof (__argp_state_help) __argp_state_help attribute_hidden;
#endif

#endif
