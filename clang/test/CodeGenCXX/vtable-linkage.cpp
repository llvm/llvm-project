// RUN: %clang_cc1 %s -triple=x86_64-pc-linux -emit-llvm -o %t
// RUN: %clang_cc1 %s -triple=x86_64-pc-linux -emit-llvm -std=c++03 -o %t.03
// RUN: %clang_cc1 %s -triple=x86_64-pc-linux -emit-llvm -std=c++11 -o %t.11
// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -disable-llvm-passes -O3 -emit-llvm -o %t.opt
// RUN: FileCheck %s < %t
// RUN: FileCheck %s < %t.03
// RUN: FileCheck %s < %t.11
// RUN: FileCheck --check-prefix=CHECK-OPT %s < %t.opt

namespace {
  struct A {
    virtual void f() { }
  };
}

void f() { A b; }

struct B {
  B();
  virtual void f();
};

B::B() { }

struct C : virtual B {
  C();
  virtual void f() { } 
};

C::C() { } 

struct D {
  virtual void f();
};

void D::f() { }

static struct : D { } e;

// Force 'e' to be constructed and therefore have a vtable defined.
void use_e() {
  e.f();
}

// The destructor is the key function.
template<typename T>
struct E {
  virtual ~E();
};

template<typename T> E<T>::~E() { }

// Anchor is the key function
template<>
struct E<char> {
  virtual void anchor();
};

void E<char>::anchor() { }

template struct E<short>;
extern template struct E<int>;

void use_E() {
  E<int> ei;
  (void)ei;
  E<long> el;
  (void)el;
}

// No key function
template<typename T>
struct F {
  virtual void foo() { }
};

// No key function
template<>
struct F<char> {
  virtual void foo() { }
};

template struct F<short>;
extern template struct F<int>;

void use_F() {
  F<char> fc;
  fc.foo();
  F<int> fi;
  fi.foo();
  F<long> fl;
  (void)fl;
}

// B has a key function that is not defined in this translation unit so its vtable
// has external linkage.
// CHECK-DAG: @_ZTV1B = external unnamed_addr constant

// C has no key function, so its vtable should have weak_odr linkage
// and hidden visibility (rdar://problem/7523229).
// CHECK-DAG: @_ZTV1C = linkonce_odr unnamed_addr constant {{.*}}, comdat, align 8{{$}}
// CHECK-DAG: @_ZTS1C = linkonce_odr constant {{.*}}, comdat, align 1{{$}}
// CHECK-DAG: @_ZTI1C = linkonce_odr constant {{.*}}, comdat, align 8{{$}}
// CHECK-DAG: @_ZTT1C = linkonce_odr unnamed_addr constant {{.*}}, comdat, align 8{{$}}

// D has a key function that is defined in this translation unit so its vtable is
// defined in the translation unit.
// CHECK-DAG: @_ZTV1D = unnamed_addr constant
// CHECK-DAG: @_ZTS1D = constant
// CHECK-DAG: @_ZTI1D = constant

// E<char> is an explicit specialization with a key function defined
// in this translation unit, so its vtable should have external
// linkage.
// CHECK-DAG: @_ZTV1EIcE = unnamed_addr constant
// CHECK-DAG: @_ZTS1EIcE = constant
// CHECK-DAG: @_ZTI1EIcE = constant

// E<short> is an explicit template instantiation with a key function
// defined in this translation unit, so its vtable should have
// weak_odr linkage.
// CHECK-DAG: @_ZTV1EIsE = weak_odr unnamed_addr constant {{.*}}, comdat,
// CHECK-DAG: @_ZTS1EIsE = weak_odr constant {{.*}}, comdat, align 1{{$}}
// CHECK-DAG: @_ZTI1EIsE = weak_odr constant {{.*}}, comdat, align 8{{$}}

// F<short> is an explicit template instantiation without a key
// function, so its vtable should have weak_odr linkage
// CHECK-DAG: @_ZTV1FIsE = weak_odr unnamed_addr constant {{.*}}, comdat,
// CHECK-DAG: @_ZTS1FIsE = weak_odr constant {{.*}}, comdat, align 1{{$}}
// CHECK-DAG: @_ZTI1FIsE = weak_odr constant {{.*}}, comdat, align 8{{$}}

// E<long> is an implicit template instantiation with a key function
// defined in this translation unit, so its vtable should have
// linkonce_odr linkage.
// CHECK-DAG: @_ZTV1EIlE = linkonce_odr unnamed_addr constant {{.*}}, comdat,
// CHECK-DAG: @_ZTS1EIlE = linkonce_odr constant {{.*}}, comdat, align 1{{$}}
// CHECK-DAG: @_ZTI1EIlE = linkonce_odr constant {{.*}}, comdat, align 8{{$}}

// F<long> is an implicit template instantiation with no key function,
// so its vtable should have linkonce_odr linkage.
// CHECK-DAG: @_ZTV1FIlE = linkonce_odr unnamed_addr constant {{.*}}, comdat,
// CHECK-DAG: @_ZTS1FIlE = linkonce_odr constant {{.*}}, comdat, align 1{{$}}
// CHECK-DAG: @_ZTI1FIlE = linkonce_odr constant {{.*}}, comdat, align 8{{$}}

// F<int> is an explicit template instantiation declaration without a
// key function, so its vtable should have external linkage.
// CHECK-DAG: @_ZTV1FIiE = external unnamed_addr constant
// CHECK-OPT-DAG: @_ZTV1FIiE = available_externally unnamed_addr constant

// E<int> is an explicit template instantiation declaration. It has a
// key function is not instantiated, so we know that vtable definition
// will be generated in TU where key function will be defined
// so we can mark it as external (without optimizations) and
// available_externally (with optimizations) because all of the inline
// virtual functions have been emitted.
// CHECK-DAG: @_ZTV1EIiE = external unnamed_addr constant
// CHECK-OPT-DAG: @_ZTV1EIiE = available_externally unnamed_addr constant

// The anonymous struct for e has no linkage, so the vtable should have
// internal linkage.
// CHECK-DAG: @"_ZTV3$_0" = internal unnamed_addr constant
// CHECK-DAG: @"_ZTS3$_0" = internal constant
// CHECK-DAG: @"_ZTI3$_0" = internal constant

// The A vtable should have internal linkage since it is inside an anonymous 
// namespace.
// CHECK-DAG: @_ZTVN12_GLOBAL__N_11AE = internal unnamed_addr constant
// CHECK-DAG: @_ZTSN12_GLOBAL__N_11AE = internal constant
// CHECK-DAG: @_ZTIN12_GLOBAL__N_11AE = internal constant

// F<char> is an explicit specialization without a key function, so
// its vtable should have linkonce_odr linkage.
// CHECK-DAG: @_ZTV1FIcE = linkonce_odr unnamed_addr constant {{.*}}, comdat,
// CHECK-DAG: @_ZTS1FIcE = linkonce_odr constant {{.*}}, comdat, align 1{{$}}
// CHECK-DAG: @_ZTI1FIcE = linkonce_odr constant {{.*}}, comdat, align 8{{$}}

// CHECK-DAG: @_ZTV1GIiE = linkonce_odr unnamed_addr constant {{.*}}, comdat,
template <typename T>
class G {
public:
  G() {}
  virtual void f0();
  virtual void f1();
};
template <>
void G<int>::f1() {}
template <typename T>
void G<T>::f0() {}
void G_f0()  { new G<int>(); }

// H<int> has a key function without a body but it's a template instantiation
// so its VTable must be emitted.
// CHECK-DAG: @_ZTV1HIiE = linkonce_odr unnamed_addr constant {{.*}}, comdat,
template <typename T>
class H {
public:
  virtual ~H();
};

void use_H() {
  H<int> h;
}

// I<int> has an explicit instantiation declaration and needs a VTT and
// construction vtables.

// CHECK-DAG: @_ZTV1IIiE = external unnamed_addr constant
// CHECK-DAG: @_ZTT1IIiE = external unnamed_addr constant
// CHECK-NOT: @_ZTC1IIiE
//
// CHECK-OPT-DAG: @_ZTV1IIiE = available_externally unnamed_addr constant
// CHECK-OPT-DAG: @_ZTT1IIiE = available_externally unnamed_addr constant
struct VBase1 { virtual void f(); }; struct VBase2 : virtual VBase1 {};
template<typename T>
struct I : VBase2 {};
extern template struct I<int>;
I<int> i;
