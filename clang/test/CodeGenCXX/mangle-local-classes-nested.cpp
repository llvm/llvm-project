// RUN: %clang_cc1 %s -emit-llvm -triple %itanium_abi_triple -o - | FileCheck %s

// CHECK: @_ZTVZZ1HvEN1S1IEvE1S = 

// CHECK: define {{.*}} @_Z2L1v(
// CHECK: define {{.*}} @_ZZ2L1vEN1S2L2Ev(
// CHECK: define {{.*}} @_ZZ2L1vEN1S2L2E_0v(
// CHECK: define {{.*}} @_ZZ1FvEN1S1T1S1T1GEv(
// CHECK: define {{.*}} @_ZZZ2L1vEN1S2L2EvEN1S3L3aEv(
// CHECK: define {{.*}} @_ZZZ2L1vEN1S2L2EvEN1S3L3bE_0v(
// CHECK: define {{.*}} @_ZZZ2L1vEN1S2L2E_0vEN1S3L3cEv(
// CHECK: define {{.*}} @_ZZZ2L1vEN1S2L2E_0vEN1S3L3dE_0v(

void L1() {
  {
    struct S {
      void L2() {
        {
          struct S {
            void L3a() {}
          };
          S().L3a();
        }
        {
          struct S {
            void L3b() {}
          };
          S().L3b();
        }
      }
    };
    S().L2();
  }
  {
    struct S {
      void L2() {
        {
          struct S {
            void L3c() {}
          };
          S().L3c();
        }
        {
          struct S {
            void L3d() {}
          };
          S().L3d();
        }
      }
    };
    S().L2();
  }
}

void F() {
  struct S {
    struct T {
      struct S {
        struct T {
          void G() {}
        };
      };
    };
  };
  S::T::S::T().G();
}

struct B { virtual void Foo() = 0; };
void G(const B &);

void H() {
  struct S {
    void I() {
      struct S : B {
        virtual void Foo() {}
      };
      G(S());
    }
  };
  S().I();
}
