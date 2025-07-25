#ifndef MULTI_CU_COMMON_H
#define MULTI_CU_COMMON_H

static inline int common_inline_function(int x) {
  int result = x * 2;
  result += 10;
  return result;
}

#endif // MULTI_CU_COMMON_H
