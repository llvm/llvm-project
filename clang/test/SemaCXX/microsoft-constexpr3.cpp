// Ignore dynamic_cast when relaxing constant expression with -fms-compatibility
// However using dynamic_cast is still possible in c++20 and higher
// RUN: not %clang_cc1 -std=c++11 -fms-compatibility -fsyntax-only %s
// RUN: %clang_cc1 -std=c++20 -fms-compatibility -fsyntax-only %s

struct B {
  virtual ~B() {}
};

struct D : B {
  int x = 123;
};

#define IsD(x) (dynamic_cast<const D*>(x) != 0)

static const D od;

constexpr bool is_d = IsD(&od);
