#include "nldbl-compat.h"

int
attribute_hidden
__printf_fp (FILE *fp, const struct printf_info *info,
	     const void *const *args)
{
  return __nldbl___printf_fp (fp, info, args);
}
