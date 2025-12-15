#include "src/__support/macros/attributes.h"

#if !LIBC_HAS_VECTOR_TYPE
#error unsupported
#endif

bool [[clang::ext_vector_type(1)]] v;
