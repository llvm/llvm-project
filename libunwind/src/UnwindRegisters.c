#include "libunwind.h"

#if defined(__NEXT__)

extern int __unw_getcontext(__attribute__((unused))
                            unw_context_t *thread_state) {
  return 0;
}

#endif
