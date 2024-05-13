#include "base.h"

#ifdef __cplusplus
int sub2(int *p) {  
  int x = p[5];
#pragma clang unsafe_buffer_usage begin
  int y = p[5];
#pragma clang unsafe_buffer_usage end
  return x + y;
}
#endif
