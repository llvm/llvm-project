// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s
template<typename T> struct A {
  virtual void f(T) { }
  inline void g() { } 
};

// Explicit instantiations have external linkage.

// CHECK: define weak_odr void @_ZN1AIiE1gEv(
template void A<int>::g();

// CHECK: define weak_odr void @_ZN1AIfE1fEf(
// CHECK: define weak_odr void @_ZN1AIfE1gEv(
// FIXME: This should also emit the vtable.
template struct A<float>;

// CHECK: define weak_odr void @_Z1fIiEvT_
template <typename T> void f(T) { }
template void f<int>(int);

// CHECK: define weak_odr void @_Z1gIiEvT_
template <typename T> inline void g(T) { }
template void g<int>(int);

template<typename T>
struct X0 {
  virtual ~X0() { }
};

template<typename T>
struct X1 : X0<T> {
  virtual void blarg();
};

template<typename T> void X1<T>::blarg() { }

extern template struct X0<char>;
extern template struct X1<char>;

// CHECK: define linkonce_odr void @_ZN2X1IcED1Ev(%struct.X1* %this) unnamed_addr
void test_X1() {
  X1<char> i1c;
}

