// RUN: %clang_cc1 -emit-llvm -triple x86_64-apple-darwin10.0.0 -o - %s | FileCheck %s
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

// CHECK-LABEL: define linkonce_odr noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_ZN1DaSERS_
// CHECK: {{call.*_ZN1AaSERS_}}
// CHECK: {{call.*_ZN1BaSERS_}}
// CHECK: {{call.*_ZN1CaSERKS_}}
// CHECK: {{call void @llvm.memcpy.p0.p0.i64.*i64 28}}
// CHECK: {{call.*_ZN1BaSERS_}}
// CHECK: br
// CHECK: {{call.*_ZN1CaSERKS_}}
// CHECK: {{call void @llvm.memcpy.p0.p0.i64.*i64 288}}
// CHECK: {{call void @llvm.memcpy.p0.p0.i64.*i64 12}}
// CHECK: call void @_ZN11CopyByValueC1ERKS_
// CHECK: {{call.*_ZN11CopyByValueaSES_}}
// CHECK: ret
