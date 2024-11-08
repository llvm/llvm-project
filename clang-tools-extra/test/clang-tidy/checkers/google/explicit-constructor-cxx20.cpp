// RUN: %check_clang_tidy %s google-explicit-constructor %t -std=c++20-or-later

namespace issue_81121
{

static constexpr bool ConstFalse = false;
static constexpr bool ConstTrue = true;

struct A {
  explicit(true) A(int);
};

struct B {
  explicit(false) B(int);
};

struct C {
  explicit(ConstTrue) C(int);
};

struct D {
  explicit(ConstFalse) D(int);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: single-argument constructors explicit expression evaluates to 'false' [google-explicit-constructor]
};

template <typename>
struct E {
  explicit(true) E(int);
};

template <typename>
struct F {
  explicit(false) F(int);
};

template <typename>
struct G {
  explicit(ConstTrue) G(int);
};

template <typename>
struct H {
  explicit(ConstFalse) H(int);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: single-argument constructors explicit expression evaluates to 'false' [google-explicit-constructor]
};

template <int Val>
struct I {
  explicit(Val > 0) I(int);
};

template <int Val>
struct J {
  explicit(Val > 0) J(int);
};

void useJ(J<0>, J<100>);

} // namespace issue_81121
