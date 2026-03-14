#include "a.h"

struct X {
  mutex m;
  int n __attribute__((guarded_by(m)));

  void f();
};

inline void unlock(X &x) __attribute__((unlock_function(x.m))) { x.m.unlock(); }
