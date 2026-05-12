// RUN: %clang_cc1 -fsyntax-only -verify -fms-compatibility -Wrelaxed-constant-fold %s

typedef long long LONG_PTR;
typedef long LONG;
#define FIELD_OFFSET(type, field) ((LONG_PTR)&(((type *)0)->field))

struct S {
  int x;
  int y;
};

constexpr long b = FIELD_OFFSET(S, y); // expected-warning {{folding this constant expression is a Microsoft extension}}
