#include <clc/utils.h>

#ifndef __CLC_FUNCTION
#define __CLC_FUNCTION(x) __CLC_CONCAT(__clc_, x)
#endif

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE FUNCTION(__CLC_GENTYPE x,
                                              private __CLC_GENTYPE *ptr) {
  return __CLC_FUNCTION(FUNCTION)(x, ptr);
}

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE FUNCTION(__CLC_GENTYPE x,
                                              global __CLC_GENTYPE *ptr) {
  return __CLC_FUNCTION(FUNCTION)(x, ptr);
}

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE FUNCTION(__CLC_GENTYPE x,
                                              local __CLC_GENTYPE *ptr) {
  return __CLC_FUNCTION(FUNCTION)(x, ptr);
}
