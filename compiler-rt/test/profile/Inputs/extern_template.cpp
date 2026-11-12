#define DEF
#include "extern_template.h"
#undef DEF
extern int bar();
extern int foo();
extern Test<int> TO;
int main() {
  foo();
  int R = bar();

  if (R != 10)
    return 1;
  return 0;
}
