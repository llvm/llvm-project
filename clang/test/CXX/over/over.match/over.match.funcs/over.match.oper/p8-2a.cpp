// RUN: %clang_cc1 -std=c++2a -verify %s

template<typename L, typename R> struct Op { L l; const char *op; R r; };
// FIXME: Remove once we implement P1816R0.
template<typename L, typename R> Op(L, R) -> Op<L, R>;

struct A {};
struct B {};
constexpr Op<A, B> operator<=>(A a, B b) { return {a, "<=>", b}; }

template<typename T, typename U, typename V> constexpr Op<Op<T, U>, V> operator<  (Op<T, U> a, V b) { return {a, "<",   b}; }
template<typename T, typename U, typename V> constexpr Op<Op<T, U>, V> operator<= (Op<T, U> a, V b) { return {a, "<=",  b}; }
template<typename T, typename U, typename V> constexpr Op<Op<T, U>, V> operator>  (Op<T, U> a, V b) { return {a, ">",   b}; }
template<typename T, typename U, typename V> constexpr Op<Op<T, U>, V> operator>= (Op<T, U> a, V b) { return {a, ">=",  b}; }
template<typename T, typename U, typename V> constexpr Op<Op<T, U>, V> operator<=>(Op<T, U> a, V b) { return {a, "<=>", b}; }

template<typename T, typename U, typename V> constexpr Op<T, Op<U, V>> operator<  (T a, Op<U, V> b) { return {a, "<",   b}; }
template<typename T, typename U, typename V> constexpr Op<T, Op<U, V>> operator<= (T a, Op<U, V> b) { return {a, "<=",  b}; }
template<typename T, typename U, typename V> constexpr Op<T, Op<U, V>> operator>  (T a, Op<U, V> b) { return {a, ">",   b}; }
template<typename T, typename U, typename V> constexpr Op<T, Op<U, V>> operator>= (T a, Op<U, V> b) { return {a, ">=",  b}; }
template<typename T, typename U, typename V> constexpr Op<T, Op<U, V>> operator<=>(T a, Op<U, V> b) { return {a, "<=>", b}; }

constexpr bool same(A, A) { return true; }
constexpr bool same(B, B) { return true; }
constexpr bool same(int a, int b) { return a == b; }
template<typename T, typename U>
constexpr bool same(Op<T, U> x, Op<T, U> y) {
  return same(x.l, y.l) && __builtin_strcmp(x.op, y.op) == 0 && same(x.r, y.r);
}

// x @ y is interpreted as:
void f(A x, B y) {
  //   --  (x <=> y) @ 0 if not reversed
  static_assert(same(x < y, (x <=> y) < 0));
  static_assert(same(x <= y, (x <=> y) <= 0));
  static_assert(same(x > y, (x <=> y) > 0));
  static_assert(same(x >= y, (x <=> y) >= 0));
  static_assert(same(x <=> y, x <=> y)); // (not rewritten)
}

void g(B x, A y) {
  //   --  0 @ (y <=> x) if reversed
  static_assert(same(x < y, 0 < (y <=> x)));
  static_assert(same(x <= y, 0 <= (y <=> x)));
  static_assert(same(x > y, 0 > (y <=> x)));
  static_assert(same(x >= y, 0 >= (y <=> x)));
  static_assert(same(x <=> y, 0 <=> (y <=> x)));
}


// We can rewrite into a call involving a builtin operator.
struct X { int result; };
struct Y {};
constexpr int operator<=>(X x, Y) { return x.result; }
static_assert(X{-1} < Y{});
static_assert(X{0} < Y{}); // expected-error {{failed}}
static_assert(X{0} <= Y{});
static_assert(X{1} <= Y{}); // expected-error {{failed}}
static_assert(X{1} > Y{});
static_assert(X{0} > Y{}); // expected-error {{failed}}
static_assert(X{0} >= Y{});
static_assert(X{-1} >= Y{}); // expected-error {{failed}}
static_assert(Y{} < X{1});
static_assert(Y{} < X{0}); // expected-error {{failed}}
static_assert(Y{} <= X{0});
static_assert(Y{} <= X{-1}); // expected-error {{failed}}
static_assert(Y{} > X{-1});
static_assert(Y{} > X{0}); // expected-error {{failed}}
static_assert(Y{} >= X{0});
static_assert(Y{} >= X{1}); // expected-error {{failed}}
