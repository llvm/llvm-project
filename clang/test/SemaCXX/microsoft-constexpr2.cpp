// RUN: %clang_cc1 -fsyntax-only -verify -fms-compatibility -Wrelaxed-constant-fold %s

typedef long long LONG_PTR;
typedef long LONG;
#define FIELD_OFFSET(type, field) ((LONG_PTR)&(((type *)0)->field))
#define FIELD_OFFSET2(type, field) (reinterpret_cast<LONG_PTR>(&(((type *)0)->field)))

struct S {
  int x;
  int y;
};

constexpr long b = FIELD_OFFSET(S, y); // expected-warning {{folding constant expression involving cast that performs the conversions of a reinterpret_cast is a Microsoft extension}}
constexpr long b2 = FIELD_OFFSET2(S, y); // expected-warning {{folding constant expression involving reinterpret_cast is a Microsoft extension}}
