// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

// PR5426 - the non-dependent obj would be fully processed and wrapped in a
// CXXConstructExpr at definition time, which would lead to a failure at
// instantiation time.
struct arg {
  arg();
};

struct oldstylemove {
  oldstylemove(oldstylemove&);
  oldstylemove(const arg&);
};

template <typename T>
void fn(T t, const arg& arg) {
  oldstylemove obj(arg);
}

void test() {
  fn(1, arg());
}

struct X0 { };

struct X1 {
  explicit X1(const X0 &x0 = X0());
};

template<typename T>
void f0() {
  X1 x1;
}

template void f0<int>();
template void f0<float>();

struct NonTrivial {
  NonTrivial();
  ~NonTrivial();
};

template<int N> void f1() {
  NonTrivial array[N];
}
template<> void f1<2>();
