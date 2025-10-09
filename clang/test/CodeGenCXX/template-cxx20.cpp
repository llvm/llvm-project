// RUN: %clang_cc1 %s -O0 -disable-llvm-passes -triple=x86_64 -std=c++20 -emit-llvm -o - | FileCheck %s

namespace GH161029_regression1 {
  template <class _Fp> auto f(int) { _Fp{}(0); }
  template <class _Fp, int... _Js> void g() {
    (..., f<_Fp>(_Js));
  }
  enum E { k };
  template <int, E> struct ElementAt;
  template <E First> struct ElementAt<0, First> {
    static int value;
  };
  template <typename T, T Item> struct TagSet {
    template <int Index> using Tag = ElementAt<Index, Item>;
  };
  template <typename TagSet> struct S {
    void U() { (void)TagSet::template Tag<0>::value; }
  };
  S<TagSet<E, k>> s;
  void h() {
    g<decltype([](auto) -> void { s.U(); }), 0>();
  }
  // CHECK: call void @_ZN20GH161029_regression11SINS_6TagSetINS_1EELS2_0EEEE1UEv
}
