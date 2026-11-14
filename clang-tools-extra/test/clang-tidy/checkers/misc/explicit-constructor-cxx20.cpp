// RUN: %check_clang_tidy %s misc-explicit-constructor %t -std=c++20-or-later

namespace issue_81121
{

static constexpr bool ConstFalse = false;
static constexpr bool ConstTrue = true;

struct A {
  explicit(true) A(int);
  template <typename T>
  explicit(true) A(T *);
  explicit(true) operator bool() const;
  template <typename T>
  explicit(true) operator T *() const;
};

struct B {
  explicit(false) B(int);
  template <typename T>
  explicit(false) B(T *);
  explicit(false) operator bool() const;
  template <typename T>
  explicit(false) operator T *() const;
};

struct C {
  explicit(ConstTrue) C(int);
  template <typename T>
  explicit(ConstTrue) C(T *);
  explicit(ConstTrue) operator bool() const;
  template <typename T>
  explicit(ConstTrue) operator T *() const;
};

struct D {
  explicit(ConstFalse) D(int);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: single-argument constructors explicit expression evaluates to 'false'
  template <typename T>
  explicit(ConstFalse) D(T *);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: single-argument constructors explicit expression evaluates to 'false'
  explicit(ConstFalse) operator bool() const;
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: 'operator bool' explicit expression evaluates to 'false'
  template <typename T>
  explicit(ConstFalse) operator T *() const;
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: 'operator type-parameter-0-0 *' explicit expression evaluates to 'false'
};

template <typename>
struct E {
  explicit(true) E(int);
  explicit(true) operator bool() const;
};

template <typename>
struct F {
  explicit(false) F(int);
  explicit(false) operator bool() const;
};

template <typename>
struct G {
  explicit(ConstTrue) G(int);
  explicit(ConstTrue) operator bool() const;
};

template <typename>
struct H {
  explicit(ConstFalse) H(int);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: single-argument constructors explicit expression evaluates to 'false'
  explicit(ConstFalse) operator bool() const;
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: 'operator bool' explicit expression evaluates to 'false'
};

template <int Val>
struct I {
  explicit(Val > 0) I(int);
  explicit(Val > 0) operator bool() const;
};

template <int Val>
struct J {
  explicit(Val > 0) J(int);
  explicit(Val > 0) operator bool() const;
};

struct K {
  template <typename T>
  explicit(sizeof(T) > 1) K(T *);
  template <typename T>
  explicit(sizeof(T) > 1) operator T *() const;
};

void useJAndK(J<0> j0, J<100> j100, K k, char *cp, int *ip) {
  bool b0 = j0;
  bool b1 = static_cast<bool>(j100);
  K k0 = cp;
  K k1(ip);
  char *p0 = k;
  int *p1 = static_cast<int *>(k);
  (void)b0;
  (void)b1;
  (void)k0;
  (void)k1;
  (void)p0;
  (void)p1;
}

} // namespace issue_81121
