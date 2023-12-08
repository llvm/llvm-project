#define USE
#include "extern_template.h"
#undef USE

Test<int> TO;
int foo() {
  TO.doIt(20);
  return TO.M;
}
