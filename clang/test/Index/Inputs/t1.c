#include "foo.h"

void foo_func(int param1) {
  int local_var = global_var;
  for (int for_var = 100; for_var < 500; ++for_var) {
    local_var = param1 + for_var;
  }
  bar_func();
}

struct S1 {
  int x;
};

struct S2 {
  int x;
};

void field_test(void) {
  struct S1 s1;
  s1.x = 0;
  ((struct S2 *)0)->x = 0;
  
  struct MyStruct ms;
  ms.field_var = 10;
}

int (^CP)(int) = ^(int x) { return x * global_var; };
