#include "macro_helper_test.h"

#define DEFINE(name) \
  namespace ns { \
  static const bool t1 = false; \
  bool t2_##name = t1; \
  bool t3_##name = t1; \
  } \
  using ns::t2_##name;

DEFINE(test)

void f1() {}
