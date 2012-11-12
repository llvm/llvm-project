// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

// CHECK: @_ZTVZN1J1KEvE1C = {{.*}} @_ZTIZN1J1KEvE1C {{.*}} @_ZZN1J1KEvENK1C1FEv 
// CHECK: @_ZTIZN1J1KEvE1C = {{.*}} @_ZTSZN1J1KEvE1C
// CHECK: @_ZTVZ1GvE1C_1 = {{.*}} @_ZTIZ1GvE1C_1 {{.*}} @_ZZ1GvENK1C1FE_1v 
// CHECK: @_ZTIZ1GvE1C_1 = {{.*}} @_ZTSZ1GvE1C_1
// CHECK: @_ZTVZ1GvE1C_0 = {{.*}} @_ZTIZ1GvE1C_0 {{.*}} @_ZZ1GvENK1C1FE_0v 
// CHECK: @_ZTIZ1GvE1C_0 = {{.*}} @_ZTSZ1GvE1C_0
// CHECK: @_ZTVZ1GvE1C = {{.*}} @_ZTIZ1GvE1C {{.*}} @_ZZ1GvENK1C1FEv 
// CHECK: @_ZTIZ1GvE1C = {{.*}} @_ZTSZ1GvE1C

// CHECK: define {{.*}} @_ZZN1J1KEvEN1CC2Ev(
// CHECK: define {{.*}} @_ZZN1J1KEvENK1C1FEv(
// CHECK: define {{.*}} @_ZZ1GvEN1CC2E_1v(
// CHECK: define {{.*}} @_ZZ1GvENK1C1FE_1v(
// CHECK: define {{.*}} @_ZZ1GvENK1C1HE_1v(
// CHECK: define {{.*}} @_ZZ1GvEN1CC2E_0v(
// CHECK: define {{.*}} @_ZZ1GvENK1C1FE_0v(
// CHECK: define {{.*}} @_ZZ1GvENK1C1GE_0v(
// CHECK: define {{.*}} @_ZZ1GvEN1CC2Ev(
// CHECK: define {{.*}} @_ZZ1GvENK1C1FEv(

struct I { 
  virtual void F() const = 0;
};

void Go(const I &i);

void G() { 
  { 
    struct C : I { 
      void F() const {}
    };
    Go(C());
  }
  { 
    struct C : I { 
      void F() const { G(); }
      void G() const {}
    };
    Go(C());
  }
  { 
    struct C : I { 
      void F() const { H(); }
      void H() const {}
    };
    Go(C());
  }
}

struct J {
  void K();
};

void J::K() {
  struct C : I {
    void F() const {}
  };
  Go(C());
}
