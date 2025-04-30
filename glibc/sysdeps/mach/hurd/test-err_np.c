#include <mach/error.h>

#define ERR_MAP(value) err_get_code (value)
#include <stdio-common/test-err_np.c>
