// RUN: %clang_cc1 %s -emit-llvm -triple %itanium_abi_triple -o - | FileCheck %s
// rdar://10246395

#define WEAK __attribute__ ((weak)) 

class WEAK A {
  virtual void foo();
};

class B : public A {
  virtual void foo();
};
void A::foo() { }
void B::foo() { }

class T {};
class T1 {};

class C : public T1, public B, public T {
  virtual void foo();
};
void C::foo() { }

class V1 : public virtual A {
  virtual void foo();
};

class V2 : public virtual V1 {
  virtual void foo();
};
void V1::foo() { }
void V2::foo() { }

// CHECK: @_ZTS1A = weak_odr constant
// CHECK: @_ZTI1A = weak_odr constant
// CHECK: @_ZTS1B = weak_odr constant
// CHECK: @_ZTI1B = weak_odr constant
// CHECK: @_ZTS1C = weak_odr constant
// CHECK: @_ZTS2T1 = linkonce_odr constant
// CHECK: @_ZTI2T1 = linkonce_odr constant
// CHECK: @_ZTS1T = linkonce_odr constant
// CHECK: @_ZTI1T = linkonce_odr constant
// CHECK: @_ZTI1C = weak_odr constant
// CHECK: @_ZTS2V1 = weak_odr constant
// CHECK: @_ZTI2V1 = weak_odr constant
// CHECK: @_ZTS2V2 = weak_odr constant
// CHECK: @_ZTI2V2 = weak_odr constant
