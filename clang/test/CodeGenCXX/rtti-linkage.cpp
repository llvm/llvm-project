// RUN: %clang_cc1 %std_cxx98- -no-opaque-pointers %s -I%S -triple=x86_64-apple-darwin10 -emit-llvm -Wno-dynamic-exception-spec -o - | FileCheck %s -check-prefixes=CHECK,CHECK-BOTH
// RUN: %clang_cc1 %std_cxx98- -no-opaque-pointers %s -I%S -triple=x86_64-apple-darwin10 -emit-llvm -Wno-dynamic-exception-spec -fvisibility=hidden -o - | FileCheck %s --check-prefixes=CHECK-WITH-HIDDEN,CHECK-BOTH

#include <typeinfo>

// CHECK-BOTH: _ZTSP1C = internal constant
// CHECK-BOTH: _ZTS1C = internal constant
// CHECK-BOTH: _ZTI1C = internal constant
// CHECK-BOTH: _ZTIP1C = internal constant
// CHECK-BOTH: _ZTSPP1C = internal constant
// CHECK-BOTH: _ZTIPP1C = internal constant
// CHECK-BOTH: _ZTSM1Ci = internal constant
// CHECK-BOTH: _ZTIM1Ci = internal constant
// CHECK-BOTH: _ZTSPM1Ci = internal constant
// CHECK-BOTH: _ZTIPM1Ci = internal constant
// CHECK-BOTH: _ZTSM1CS_ = internal constant
// CHECK-BOTH: _ZTIM1CS_ = internal constant
// CHECK-BOTH: _ZTSM1CPS_ = internal constant
// CHECK-BOTH: _ZTIM1CPS_ = internal constant
// CHECK-BOTH: _ZTSM1A1C = internal constant
// CHECK: _ZTS1A = linkonce_odr constant
// CHECK-WITH-HIDDEN: _ZTS1A = linkonce_odr hidden constant
// CHECK: _ZTI1A = linkonce_odr constant
// CHECK-WITH-HIDDEN: _ZTI1A = linkonce_odr hidden constant
// CHECK-BOTH: _ZTIM1A1C = internal constant
// CHECK-BOTH: _ZTSM1AP1C = internal constant
// CHECK-BOTH: _ZTIM1AP1C = internal constant

// CHECK-WITH-HIDDEN: _ZTSFN12_GLOBAL__N_11DEvE = internal constant
// CHECK-WITH-HIDDEN: @_ZTSPK2T4 = linkonce_odr hidden constant 
// CHECK-WITH-HIDDEN: @_ZTS2T4 = linkonce_odr hidden constant 
// CHECK-WITH-HIDDEN: @_ZTI2T4 = linkonce_odr hidden constant 
// CHECK-WITH-HIDDEN: @_ZTIPK2T4 = linkonce_odr hidden constant 
// CHECK-WITH-HIDDEN: @_ZTSZ2t5vE1A = internal constant
// CHECK-WITH-HIDDEN: @_ZTIZ2t5vE1A = internal constant
// CHECK-WITH-HIDDEN: @_ZTSZ2t6vE1A = linkonce_odr hidden constant
// CHECK-WITH-HIDDEN: @_ZTIZ2t6vE1A = linkonce_odr hidden constant
// CHECK-WITH-HIDDEN: @_ZTSPZ2t7vE1A = linkonce_odr hidden constant
// CHECK-WITH-HIDDEN: @_ZTSZ2t7vE1A = linkonce_odr hidden constant
// CHECK-WITH-HIDDEN: @_ZTIZ2t7vE1A = linkonce_odr hidden constant
// CHECK-WITH-HIDDEN: @_ZTIPZ2t7vE1A = linkonce_odr hidden constant

// CHECK: _ZTSN12_GLOBAL__N_11DE = internal constant
// CHECK: _ZTIN12_GLOBAL__N_11DE = internal constant
// CHECK: _ZTSPN12_GLOBAL__N_11DE = internal constant
// CHECK: _ZTIPN12_GLOBAL__N_11DE = internal constant
// CHECK: _ZTSFN12_GLOBAL__N_11DEvE = internal constant
// CHECK: _ZTIFN12_GLOBAL__N_11DEvE = internal constant
// CHECK: _ZTSFvN12_GLOBAL__N_11DEE = internal constant
// CHECK: _ZTIFvN12_GLOBAL__N_11DEE = internal constant
// CHECK: _ZTSPFvvE = linkonce_odr constant
// CHECK: _ZTSFvvE = linkonce_odr constant
// CHECK: _ZTIFvvE = linkonce_odr constant
// CHECK: _ZTIPFvvE = linkonce_odr constant
// CHECK: _ZTSN12_GLOBAL__N_11EE = internal constant
// CHECK: _ZTIN12_GLOBAL__N_11EE = internal constant
// CHECK: _ZTSA10_i = linkonce_odr constant
// CHECK: _ZTIA10_i = linkonce_odr constant
// CHECK: _ZTI1TILj0EE = linkonce_odr constant
// CHECK: _ZTI1TILj1EE = weak_odr constant
// CHECK: _ZTI1TILj2EE = external constant
// CHECK: _ZTSZ2t5vE1A = internal constant
// CHECK: _ZTIZ2t5vE1A = internal constant
// CHECK: _ZTS1B ={{.*}} constant
// CHECK: _ZTI1B ={{.*}} constant
// CHECK: _ZTS1F = linkonce_odr constant
// CHECK: _ZTSZ2t6vE1A = linkonce_odr constant
// CHECK: _ZTIZ2t6vE1A = linkonce_odr constant
// CHECK: _ZTSPZ2t7vE1A = linkonce_odr constant
// CHECK: _ZTSZ2t7vE1A = linkonce_odr constant
// CHECK: _ZTIZ2t7vE1A = linkonce_odr constant
// CHECK: _ZTIPZ2t7vE1A = linkonce_odr constant

// CHECK: _ZTIN12_GLOBAL__N_11DE to

// A has no key function, so its RTTI data should be linkonce_odr.
struct A { };

// B has a key function defined in the translation unit, so the RTTI data should
// be emitted in this translation unit and have external linkage.
struct B : A {
  virtual void f();
};
void B::f() { }

// C is an incomplete class type, so any direct or indirect pointer types should have 
// internal linkage, as should the type info for C itself.
struct C;

void t1() {
  (void)typeid(C*);
  (void)typeid(C**);
  (void)typeid(int C::*);
  (void)typeid(int C::**);
  (void)typeid(C C::*);
  (void)typeid(C *C::*);
  (void)typeid(C A::*);
  (void)typeid(C* A::*);
}

namespace {
  // D is inside an anonymous namespace, so all type information related to D should have
  // internal linkage.
  struct D { };
  
  // E is also inside an anonymous namespace.
  enum E { };
  
};

// F has a key function defined in the translation unit, but it is inline so the RTTI
// data should be emitted with linkonce_odr linkage.
struct F {
  virtual void f();
};

inline void F::f() { }
const D getD();

const std::type_info &t2() {
  (void)typeid(const D);
  (void)typeid(D *);
  (void)typeid(D (*)());
  (void)typeid(void (*)(D));
  (void)typeid(void (*)(D&));
  // The exception specification is not part of the RTTI descriptor, so it should not have
  // internal linkage.
  (void)typeid(void (*)() throw (D));
  
  (void)typeid(E);
  
  return typeid(getD());  
}

namespace Arrays {
  struct A {
    static const int a[10];
  };
  const std::type_info &f() {
    return typeid(A::a);
  }
}

template <unsigned N> class T {
  virtual void anchor() {}
};
template class T<1>;
template <> class T<2> { virtual void anchor(); };
void t3() {
  (void) typeid(T<0>);
  (void) typeid(T<1>);
  (void) typeid(T<2>);
}

// rdar://problem/8778973
struct T4 {};
void t4(const T4 *ptr) {
  const void *value = &typeid(ptr);
}

// rdar://16265084
void t5() {
  struct A {};
  const void *value = &typeid(A);
}

inline void t6() {
  struct A {};
  const void *value = &typeid(A);
}
void t6_helper() {
  t6();
}

inline void t7() {
  struct A {};
  const void *value = &typeid(A*);
}
void t7_helper() {
  t7();
}
