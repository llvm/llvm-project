// RUN: %clang_cc1 -emit-llvm -triple i686-pc-linux-gnu -o - %s | FileCheck %s

// This check logically is attached to 'template int S<int>::i;' below.
// CHECK: @_ZN1SIiE1iE = weak_odr global i32

template<typename T, typename U, typename Result>
struct plus {
  Result operator()(const T& t, const U& u) const;
};

template<typename T, typename U, typename Result>
Result plus<T, U, Result>::operator()(const T& t, const U& u) const {
  return t + u;
}

// CHECK: define weak_odr i32 @_ZNK4plusIillEclERKiRKl
template struct plus<int, long, long>;

// Check that we emit definitions from explicit instantiations even when they
// occur prior to the definition itself.
template <typename T> struct S {
  void f();
  static void g();
  static int i;
  struct S2 {
    void h();
  };
};

// CHECK: define weak_odr void @_ZN1SIiE1fEv
template void S<int>::f();

// CHECK: define weak_odr void @_ZN1SIiE1gEv
template void S<int>::g();

// See the check line at the top of the file.
template int S<int>::i;

// CHECK: define weak_odr void @_ZN1SIiE2S21hEv
template void S<int>::S2::h();

template <typename T> void S<T>::f() {}
template <typename T> void S<T>::g() {}
template <typename T> int S<T>::i;
template <typename T> void S<T>::S2::h() {}
