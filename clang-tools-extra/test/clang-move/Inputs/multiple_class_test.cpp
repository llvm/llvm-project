#include "multiple_class_test.h"

using a::Move1;
using namespace a;
using A = a::Move1;
static int g = 0;

namespace a {
int Move1::f() {
  return 0;
}
} // namespace a

namespace {
using a::Move1;
using namespace a;
static int k = 0;
} // namespace

namespace b {
using a::Move1;
using namespace a;
using T = a::Move1;
int Move2::f() {
  return 0;
}
} // namespace b

namespace c {
int Move3::f() {
  using a::Move1;
  using namespace b;
  return 0;
}

int Move4::f() {
  return k;
}

int EnclosingMove5::a = 1;

int EnclosingMove5::Nested::f() {
  return g;
}

int EnclosingMove5::Nested::b = 1;

int NoMove::f() {
  static int F = 0;
  return g;
}
} // namespace c
