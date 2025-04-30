/* Error handling in libm-style for libc.  */

#include <errno.h>

#include "libm_support.h"


void
__libm_error_support (void *arg1, void *arg2, void *retval,
		      error_types input_tag)
{
  __set_errno (ERANGE);
}
libc_hidden_def (__libm_error_support)
