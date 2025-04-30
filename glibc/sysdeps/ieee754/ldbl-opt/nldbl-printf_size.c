#include "nldbl-compat.h"

int
attribute_hidden
printf_size (FILE *__restrict fp, const struct printf_info *info,
	     const void *const *__restrict args)
{
  return __nldbl_printf_size (fp, info, args);
}
