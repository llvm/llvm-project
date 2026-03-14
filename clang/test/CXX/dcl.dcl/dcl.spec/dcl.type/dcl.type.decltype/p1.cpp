// RUN: %clang_cc1 -std=c++20 -verify %s
// expected-no-diagnostics

// bullet 2: decltype(x) where x is a non-type template parameter gives the
// type of X, after deduction, if any.
namespace ClassNTTP {
  template<decltype(auto) v, typename ParamT, typename ExprT> void f() {
    using U = decltype(v);
    using U = ParamT;

    using V = decltype((v));
    using V = ExprT;
  }

  // The names of most non-reference NTTPs are prvalues.
  template void f<0, int, int>();

  // The name of a class NTTP of type T is an lvalue of type 'const T'.
  struct X {};
  template void f<X{}, X, const X&>();

  // Ensure we get this right for references to classes too.
  template<auto x> auto &TempParamObject = x;
  template void f<TempParamObject<X{}>, const X&, const X&>();

  struct Y {} y;
  template void f<(y), Y&, Y&>();
  template void f<y, Y, const Y&>();
}
