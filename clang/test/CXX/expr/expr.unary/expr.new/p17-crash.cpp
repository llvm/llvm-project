// RUN: %clang_cc1 -emit-llvm-only %s

// this used to crash due to templ<int>'s dtor not being marked as used by the
// new expression in func()
struct non_trivial {
  non_trivial() {} 
  ~non_trivial() {}
};
template < typename T > class templ {
  non_trivial n;
};
void func() {
  new templ<int>[1][1];
}
