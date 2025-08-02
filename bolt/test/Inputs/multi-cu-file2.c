#include "multi-cu-common.h"
#include <stdio.h>

void helper_function() {
  int value = 10;
  int result = common_inline_function(value);
  printf("File2: Helper result is %d\n", result);
}
