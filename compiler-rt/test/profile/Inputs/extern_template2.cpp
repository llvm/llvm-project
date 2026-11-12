#define USE
#include "extern_template.h"
#undef USE

extern Test<int> TO;
int bar() {
  TO.doIt(5);
  return TO.M;
}
