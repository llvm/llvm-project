#include "system.h"
#define USER_NOP() do {} while (SYSTEM_FALSE)
#define USER_NOP_ALIAS() do {} while (SYSTEM_FALSE_ALIAS)
#define USER_ASSERT(expr) USER_NOP_ALIAS()
#define DECLARE_COMMON(c) \
  static inline c *c##_cast(void *p) { \
    USER_ASSERT(p); \
    return (c *)p; \
  }
#define DECLARE_CLASS(c) DECLARE_COMMON(c)
