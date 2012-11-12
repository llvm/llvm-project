// RUN: %clang_cc1 -fobjc-gc -emit-llvm -triple x86_64-apple-darwin10.0.0 -fobjc-runtime=macosx-fragile-10.5 -o - %s | FileCheck %s -check-prefix=CHECK-OBJ
// RUN: %clang_cc1 -x c++    -emit-llvm -triple x86_64-apple-darwin10.0.0                                    -o - %s | FileCheck %s -check-prefix=CHECK-CPP
#ifdef __OBJC__
struct A { 
  A &operator=(const A&);
  A &operator=(A&);
};

struct B {
  B &operator=(B&);
};

struct C {
  virtual C& operator=(const C&);
};

struct POD {
  id myobjc;
  int array[3][4];
};

struct CopyByValue {
  CopyByValue(const CopyByValue&);
  CopyByValue &operator=(CopyByValue);
};

struct D : A, B, virtual C { 
  int scalar;
  int scalar_array[2][3];
  B class_member;
  C class_member_array[2][3];
  POD pod_array[2][3];

  union {
    int x;
    float f[3];
  };

  CopyByValue by_value;
};

void test_D(D d1, D d2) {
  d1 = d2;
}

// CHECK-OBJ: define linkonce_odr %struct.D* @_ZN1DaSERS_
// CHECK-OBJ: {{call.*_ZN1AaSERS_}}
// CHECK-OBJ: {{call.*_ZN1BaSERS_}}
// CHECK-OBJ: {{call.*_ZN1CaSERKS_}}
// CHECK-OBJ: {{call void @llvm.memcpy.p0i8.p0i8.i64.*i64 24}}
// CHECK-OBJ: {{call.*_ZN1BaSERS_}}
// CHECK-OBJ: br
// CHECK-OBJ: {{call.*_ZN1CaSERKS_}}
// CHECK-OBJ: {{call.*@objc_memmove_collectable}}
// CHECK-OBJ: {{call void @llvm.memcpy.p0i8.p0i8.i64.*i64 12}}
// CHECK-OBJ: call void @_ZN11CopyByValueC1ERKS_
// CHECK-OBJ: {{call.*_ZN11CopyByValueaSES_}}
// CHECK-OBJ: ret
#endif

namespace PR13329 {
#ifndef __OBJC__
  typedef void* id;
#endif
  struct POD {
    id    i;
    short s;
  };
  
  struct NonPOD {
    id    i;
    short s;
    
    NonPOD();
  };
  
  struct DerivedNonPOD: NonPOD {
    char  c;
  };
  
  struct DerivedPOD: POD {
    char  c;
  };
  
  void testPOD() {
    POD a;
    POD b;
    // CHECK-OBJ: @objc_memmove_collectable{{.*}}i64 16
    // CHECK-CPP: @llvm.memcpy{{.*}}i64 16
    b = a;
  }
  
  void testNonPOD() {
    NonPOD a;
    NonPOD b;
    // CHECK-OBJ: @objc_memmove_collectable{{.*}}i64 10
    // CHECK-CPP: @llvm.memcpy{{.*}}i64 10
    b = a;
  }
  
  void testDerivedNonPOD() {
    DerivedNonPOD a;
    NonPOD        b;
    DerivedNonPOD c;
    // CHECK-OBJ: @objc_memmove_collectable{{.*}}i64 10
    // CHECK-CPP: @llvm.memcpy{{.*}}i64 10
    (NonPOD&) a = b;
    // CHECK-OBJ: @objc_memmove_collectable{{.*}}i64 11
    // CHECK-CPP: @llvm.memcpy{{.*}}i64 11
    a = c;
  };
  
  void testDerivedPOD() {
    DerivedPOD a;
    POD        b;
    DerivedPOD c;
    // CHECK-OBJ: @objc_memmove_collectable{{.*}}i64 16
    // CHECK-CPP: @llvm.memcpy{{.*}}i64 16
    (POD&) a = b;
    // CHECK-OBJ: @objc_memmove_collectable{{.*}}i64 17
    // CHECK-CPP: @llvm.memcpy{{.*}}i64 17
    a = c;
  };
}
