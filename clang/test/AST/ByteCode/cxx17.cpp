// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -std=c++17 -verify=expected,both %s
// RUN: %clang_cc1 -std=c++17 -verify=ref,both %s

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

namespace cwg1872 {
  template<typename T> struct A : T {
    constexpr int f() const { return 0; }
  };
  struct X {};
  struct Y { virtual int f() const; };
  struct Z : virtual X {};

  constexpr int z = A<Z>().f(); // both-error {{must be initialized by a constant expression}} \
                                // both-note {{non-literal type 'A<Z>' cannot be used in a constant expression}}
}

/// The diagnostics between the two interpreters used to be different here.
struct S { int a; };
constexpr S getS() { // both-error {{constexpr function never produces a constant expression}}
  (void)(1/0); // both-note 2{{division by zero}} \
               // both-warning {{division by zero}}
  return S{12};
}
constexpr S s = getS(); // both-error {{must be initialized by a constant expression}} \
                        // both-note {{in call to 'getS()'}} \
                        // both-note {{declared here}}
static_assert(s.a == 12, ""); // both-error {{not an integral constant expression}} \
                              // both-note {{initializer of 's' is not a constant expression}}

using size_t = decltype(sizeof(0));
namespace std { template<typename T> struct tuple_size; }
namespace std { template<size_t, typename> struct tuple_element; }

namespace constant {
  struct Q {};
  template<int N> constexpr int get(Q &&) { return N * N; }
}
template<> struct std::tuple_size<constant::Q> { static const int value = 3; };
template<int N> struct std::tuple_element<N, constant::Q> { typedef int type; };

namespace constant {
  Q q;
  constexpr bool f() {
    auto [a, b, c] = q;
    return a == 0 && b == 1 && c == 4;
  }
  static_assert(f());
}
