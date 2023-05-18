#include "a.h"

struct X {
  mutex m;
  int n __attribute__((guarded_by(m)));

  void f();
};
