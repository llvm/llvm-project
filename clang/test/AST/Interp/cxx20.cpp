// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -std=c++20 -verify %s
// RUN: %clang_cc1 -std=c++20 -verify=ref %s


constexpr int getMinus5() {
  int a = 10;
  a = -5;
  int *p = &a;
  return *p;
}
static_assert(getMinus5() == -5, "");

constexpr int assign() {
  int m = 10;
  int k = 12;

  m = (k = 20);

  return m;
}
static_assert(assign() == 20, "");


constexpr int pointerAssign() {
  int m = 10;
  int *p = &m;

  *p = 12; // modifies m

  return m;
}
static_assert(pointerAssign() == 12, "");

constexpr int pointerDeref() {
  int m = 12;
  int *p = &m;

  return *p;
}
static_assert(pointerDeref() == 12, "");

constexpr int pointerAssign2() {
  int m = 10;
  int *p = &m;
  int **pp = &p;

  **pp = 12;

  int v = **pp;

  return v;
}
static_assert(pointerAssign2() == 12, "");


constexpr int unInitLocal() {
  int a;
  return a; // ref-note{{read of uninitialized object}}
}
static_assert(unInitLocal() == 0, ""); // expected-error {{not an integral constant expression}} \
                                       // ref-error {{not an integral constant expression}} \
                                       // ref-note {{in call to 'unInitLocal()'}}

/// TODO: The example above is correctly rejected by the new constexpr
///   interpreter, but for the wrong reasons. We don't reject it because
///   it is an uninitialized read, we reject it simply because
///   the local variable does not have an initializer.
///
///   The code below should be accepted but is also being rejected
///   right now.
#if 0
constexpr int initializedLocal() {
  int a;
  int b;

  a = 20;
  return a;
}
static_assert(initializedLocal() == 20);
#endif
