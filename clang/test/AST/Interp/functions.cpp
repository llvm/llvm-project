// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify %s
// RUN: %clang_cc1 -verify=ref %s

// expected-no-diagnostics
// ref-no-diagnostics

constexpr void doNothing() {}
constexpr int gimme5() {
  doNothing();
  return 5;
}
static_assert(gimme5() == 5, "");


template<typename T> constexpr T identity(T t) { return t; }
static_assert(identity(true), "");
static_assert(identity(true), ""); /// Compiled bytecode should be cached
static_assert(!identity(false), "");

constexpr auto add(int a, int b) -> int {
  return identity(a) + identity(b);
}

constexpr int sub(int a, int b) {
  return a - b;
}
static_assert(sub(5, 2) == 3, "");
static_assert(sub(0, 5) == -5, "");

constexpr int norm(int n) {
  if (n >= 0) {
    return identity(n);
  }
  return -identity(n);
}
static_assert(norm(5) == norm(-5), "");

constexpr int square(int n) {
  return norm(n) * norm(n);
}
static_assert(square(2) == 4, "");

constexpr int add_second(int a, int b, bool doAdd = true) {
  if (doAdd)
    return a + b;
  return a;
}
static_assert(add_second(10, 3, true) == 13, "");
static_assert(add_second(10, 3) == 13, "");
static_assert(add_second(300, -20, false) == 300, "");


constexpr int sub(int a, int b, int c) {
  return a - b - c;
}
static_assert(sub(10, 8, 2) == 0, "");


constexpr int recursion(int i) {
  doNothing();
  i = i - 1;
  if (i == 0)
    return identity(0);

  return recursion(i);
}
static_assert(recursion(10) == 0, "");

template<int N = 5>
constexpr decltype(N) getNum() {
  return N;
}
static_assert(getNum<-2>() == -2, "");
static_assert(getNum<10>() == 10, "");
static_assert(getNum() == 5, "");
