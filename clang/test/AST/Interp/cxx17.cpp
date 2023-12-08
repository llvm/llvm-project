// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -std=c++17 -verify %s
// RUN: %clang_cc1 -std=c++17 -verify=ref %s

// ref-no-diagnostics
// expected-no-diagnostics

struct F { int a; int b;};
constexpr F getF() {
  return {12, 3};
}
constexpr int f() {
  auto [a1, b1] = getF();
  auto [a2, b2] = getF();

  return a1 + a2 + b1 + b2;
}
static_assert(f() == 30);


constexpr int structRefs() {
  const auto &[a, b] = getF();

  return a + b;
}
static_assert(structRefs() == 15);

constexpr int structRefs2() {
  F f = getF();
  const auto &[a, b] = f;

  return a + b;
}
static_assert(structRefs2() == 15);


template<typename T>
struct Tuple {
  T first;
  T second;
  constexpr Tuple(T a, T b) : first(a), second(b) {}
};
template<typename T>
constexpr T addTuple(const Tuple<T> &Tu) {
  auto [a, b] = Tu;
  return a + b;
}

template<typename T>
constexpr T addTuple2(const Tuple<T> &Tu) {
  auto [a, b] = Tu;
  return Tu.first + Tu.second;
}

constexpr Tuple<int> T1 = Tuple(1,2);
static_assert(addTuple(T1) == 3);
static_assert(addTuple2(T1) == 3);

constexpr Tuple<short> T2 = Tuple<short>(11,2);
static_assert(addTuple(T2) == 13);
static_assert(addTuple2(T2) == 13);

constexpr int Modify() {
  auto T = Tuple<int>(10, 20);
  auto &[x, y] = T;
  x += 1;
  y += 1;
  return T.first + T.second;
}
static_assert(Modify() == 32, "");

constexpr int a() {
  int a[2] = {5, 3};
  auto [x, y] = a;
  return x + y;
}
static_assert(a() == 8);

constexpr int b() {
  int a[2] = {5, 3};
  auto &[x, y] = a;
  x += 1;
  y += 2;
  return a[0] + a[1];
}
static_assert(b() == 11);
