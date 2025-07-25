#include "multi-cu-common.h"
#include <stdio.h>

int main() {
  int value = 5;
  int result = common_inline_function(value);
  printf("File1: Result is %d\n", result);
  return 0;
}
