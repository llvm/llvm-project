#include "include-nested.h"

void kernel nested1(__global int *j) {
  *j += 2;
  nested2(j);
}
