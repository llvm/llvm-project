template<typename T> struct is_floating {
  enum { value = 0 };
  typedef int type;
};
#include "b.h"
bool n20 = is_floating<int>::value;
