#include "std-compare.h"

struct A {
  int a;
  constexpr auto operator<=>(const A&) const = default;
};

bool foo(A x, A y) { return x < y; }
