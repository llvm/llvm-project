#include "foo.h"

int global_var = 10;

void bar_func(void) {
  global_var += 100;
  foo_func(global_var);

  struct MyStruct *ms;
  ms->field_var = 10;
}
