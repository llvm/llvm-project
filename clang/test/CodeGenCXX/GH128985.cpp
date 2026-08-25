// RUN: %clang_cc1 %s -triple x86_64 --embed-dir=%S/Inputs -emit-llvm -o - | FileCheck %s

// embed-data.txt contains "0123456789" (48 ... 57) without a trailing newline.

struct S {
  int a, b;
};

struct A {
  A(char);
};

// CHECK-LABEL: define {{.*}}void @_Z2f1i(
// CHECK: icmp ult i64 %{{.*}}, 4
// CHECK: %[[A1:.*]] = call {{.*}}ptr @_Znam(i64 {{.*}})
// CHECK: store i32 48, ptr %[[A1]]
// CHECK: %[[F1E1:.*]] = getelementptr inbounds i32, ptr %[[A1]], i64 1
// CHECK: store i32 49, ptr %[[F1E1]]
// CHECK: %[[F1E2:.*]] = getelementptr inbounds i32, ptr %[[F1E1]], i64 1
// CHECK: store i32 50, ptr %[[F1E2]]
// CHECK: %[[F1E3:.*]] = getelementptr inbounds i32, ptr %[[F1E2]], i64 1
// CHECK: store i32 51, ptr %[[F1E3]]
// CHECK: %[[F1REST:.*]] = sub i64 %{{.*}}, 16
// CHECK: call void @llvm.memset.p0.i64(ptr align 4 %{{.*}}, i8 0, i64 %[[F1REST]], i1 false)
void f1(int x) {
  int *p = new int[x]{
#embed <embed-data.txt> limit(4)
  };
}

// CHECK-LABEL: define {{.*}}void @_Z2f2i(
// CHECK: icmp ult i64 %{{.*}}, 4
// CHECK: %[[A2:.*]] = call {{.*}}ptr @_Znam(i64 {{.*}})
// CHECK: store i32 500, ptr %[[A2]]
// CHECK: %[[F2E1:.*]] = getelementptr inbounds i32, ptr %[[A2]], i64 1
// CHECK: store i32 48, ptr %[[F2E1]]
// CHECK: %[[F2E2:.*]] = getelementptr inbounds i32, ptr %[[F2E1]], i64 1
// CHECK: store i32 49, ptr %[[F2E2]]
// CHECK: %[[F2E3:.*]] = getelementptr inbounds i32, ptr %[[F2E2]], i64 1
// CHECK: store i32 600, ptr %[[F2E3]]
// CHECK: %[[F2REST:.*]] = sub i64 %{{.*}}, 16
// CHECK: call void @llvm.memset.p0.i64(ptr align 4 %{{.*}}, i8 0, i64 %[[F2REST]], i1 false)
void f2(int x) {
  int *p = new int[x]{
    500,
#embed <embed-data.txt> limit(2) suffix(, 600)
  };
}

// char arrays go through the string literal initialization path.
// CHECK-LABEL: define {{.*}}void @_Z2f3i(
// CHECK: icmp ult i64 %{{.*}}, 4
// CHECK: %[[A3:.*]] = call {{.*}}ptr @_Znam(i64 {{.*}})
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 1 %[[A3]], ptr align 1 @{{.*}}, i64 4, i1 false)
// CHECK: %[[F3END:.*]] = getelementptr inbounds i8, ptr %[[A3]], i64 4
// CHECK: %[[F3REST:.*]] = sub i64 %{{.*}}, 4
// CHECK: call void @llvm.memset.p0.i64(ptr align 1 %[[F3END]], i8 0, i64 %[[F3REST]], i1 false)
void f3(int x) {
  char *p = new char[x]{
#embed <embed-data.txt> limit(4)
  };
}

// CHECK-LABEL: define {{.*}}void @_Z2f4i(
// CHECK: icmp ult i64 %{{.*}}, 2
// CHECK: %[[A4:.*]] = call {{.*}}ptr @_Znam(i64 {{.*}})
// CHECK: store i32 900, ptr %[[A4]]
// CHECK: %[[F4E1:.*]] = getelementptr inbounds i32, ptr %[[A4]], i64 1
// CHECK: store i32 48, ptr %[[F4E1]]
// CHECK: %[[F4REST:.*]] = sub i64 %{{.*}}, 8
// CHECK: call void @llvm.memset.p0.i64(ptr align 4 %{{.*}}, i8 0, i64 %[[F4REST]], i1 false)
void f4(int x) {
  int *p = new int[x]{
#embed <embed-data.txt> limit(1) prefix(900, )
  };
}

// Constant size fully covered by the embed data: no trailing fill.
// CHECK-LABEL: define {{.*}}void @_Z2f5v(
// CHECK: %[[A5:.*]] = call {{.*}}ptr @_Znam(i64 {{.*}})
// CHECK: store i32 48, ptr %[[A5]]
// CHECK: %[[F5E1:.*]] = getelementptr inbounds i32, ptr %[[A5]], i64 1
// CHECK: store i32 49, ptr %[[F5E1]]
// CHECK: %[[F5E2:.*]] = getelementptr inbounds i32, ptr %[[F5E1]], i64 1
// CHECK: store i32 50, ptr %[[F5E2]]
// CHECK: %[[F5E3:.*]] = getelementptr inbounds i32, ptr %[[F5E2]], i64 1
// CHECK: store i32 51, ptr %[[F5E3]]
// CHECK-NOT: call void @llvm.memset
// CHECK: ret void
void f5() {
  int *p = new int[4]{
#embed <embed-data.txt> limit(4)
  };
}

// Sema wraps the EmbedExpr in an implicit conversion to the element type.
// CHECK-LABEL: define {{.*}}void @_Z2f6i(
// CHECK: icmp ult i64 %{{.*}}, 4
// CHECK: %[[A6:.*]] = call {{.*}}ptr @_Znam(i64 {{.*}})
// CHECK: store i64 48, ptr %[[A6]]
// CHECK: %[[F6E1:.*]] = getelementptr inbounds i64, ptr %[[A6]], i64 1
// CHECK: store i64 49, ptr %[[F6E1]]
// CHECK: %[[F6E2:.*]] = getelementptr inbounds i64, ptr %[[F6E1]], i64 1
// CHECK: store i64 50, ptr %[[F6E2]]
// CHECK: %[[F6E3:.*]] = getelementptr inbounds i64, ptr %[[F6E2]], i64 1
// CHECK: store i64 51, ptr %[[F6E3]]
// CHECK: %[[F6REST:.*]] = sub i64 %{{.*}}, 32
// CHECK: call void @llvm.memset.p0.i64(ptr align 8 %{{.*}}, i8 0, i64 %[[F6REST]], i1 false)
void f6(int x) {
  long long *p = new long long[x]{
#embed <embed-data.txt> limit(4)
  };
}

// CHECK-LABEL: define {{.*}}void @_Z2f7i(
// CHECK: icmp ult i64 %{{.*}}, 2
// CHECK: call {{.*}}ptr @_Znam(i64 {{.*}})
// CHECK: store i32 48, ptr
// CHECK: store i32 49, ptr
// CHECK: store i32 50, ptr
// CHECK: store i32 51, ptr
// CHECK: %[[F7REST:.*]] = sub i64 %{{.*}}, 16
// CHECK: call void @llvm.memset.p0.i64(ptr align {{[0-9]+}} %{{.*}}, i8 0, i64 %[[F7REST]], i1 false)
void f7(int x) {
  int (*p)[2] = new int[x][2]{
#embed <embed-data.txt> limit(4)
  };
}

// CHECK-LABEL: define {{.*}}void @_Z2f8i(
// CHECK: icmp ult i64 %{{.*}}, 2
// CHECK: call {{.*}}ptr @_Znam(i64 {{.*}})
// CHECK: store i32 48, ptr
// CHECK: store i32 49, ptr
// CHECK: store i32 50, ptr
// CHECK: store i32 51, ptr
// CHECK: %[[F8REST:.*]] = sub i64 %{{.*}}, 16
// CHECK: call void @llvm.memset.p0.i64(ptr align {{[0-9]+}} %{{.*}}, i8 0, i64 %[[F8REST]], i1 false)
void f8(int x) {
  S *p = new S[x]{
#embed <embed-data.txt> limit(4)
  };
}

// CHECK-LABEL: define {{.*}}void @_Z2f9i(
// CHECK: icmp ult i64 %{{.*}}, 2
// CHECK: call {{.*}}ptr @_Znam(i64 {{.*}})
// CHECK: store i32 48, ptr
// CHECK: store i32 49, ptr
// CHECK: store i32 50, ptr
// CHECK: store i32 51, ptr
// CHECK: store i32 52, ptr
// CHECK: store i32 53, ptr
// CHECK: store i32 54, ptr
// CHECK: store i32 55, ptr
// CHECK: %[[F9REST:.*]] = sub i64 %{{.*}}, 32
// CHECK: call void @llvm.memset.p0.i64(ptr align {{[0-9]+}} %{{.*}}, i8 0, i64 %[[F9REST]], i1 false)
void f9(int x) {
  S (*p)[2] = new S[x][2]{
#embed <embed-data.txt> limit(8)
  };
}

// CHECK-LABEL: define {{.*}}void @_Z3f10v(
// CHECK: %[[A10:.*]] = call {{.*}}ptr @_Znam(i64 noundef 16)
// CHECK: store i32 48, ptr %[[A10]]
// CHECK: %[[F10E1:.*]] = getelementptr inbounds i32, ptr %[[A10]], i64 1
// CHECK: store i32 49, ptr %[[F10E1]]
// CHECK: %[[F10E2:.*]] = getelementptr inbounds i32, ptr %[[F10E1]], i64 1
// CHECK: store i32 50, ptr %[[F10E2]]
// CHECK: %[[F10E3:.*]] = getelementptr inbounds i32, ptr %[[F10E2]], i64 1
// CHECK: store i32 51, ptr %[[F10E3]]
// CHECK-NOT: call void @llvm.memset
// CHECK: ret void
void f10() {
  int *p = new int[]{
#embed <embed-data.txt> limit(4)
  };
}

// CHECK-LABEL: define {{.*}}void @_Z3f11i(
// CHECK: icmp ult i64 %{{.*}}, 4
// CHECK: %[[A11:.*]] = call {{.*}}ptr @_Znam(i64 {{.*}})
// CHECK: store float 4.800000e+01, ptr %[[A11]]
// CHECK: %[[F11E1:.*]] = getelementptr inbounds float, ptr %[[A11]], i64 1
// CHECK: store float 4.900000e+01, ptr %[[F11E1]]
// CHECK: %[[F11E2:.*]] = getelementptr inbounds float, ptr %[[F11E1]], i64 1
// CHECK: store float 5.000000e+01, ptr %[[F11E2]]
// CHECK: %[[F11E3:.*]] = getelementptr inbounds float, ptr %[[F11E2]], i64 1
// CHECK: store float 5.100000e+01, ptr %[[F11E3]]
// CHECK: %[[F11REST:.*]] = sub i64 %{{.*}}, 16
// CHECK: call void @llvm.memset.p0.i64(ptr align 4 %{{.*}}, i8 0, i64 %[[F11REST]], i1 false)
void f11(int x) {
  float *p = new float[x]{
#embed <embed-data.txt> limit(4)
  };
}

// Class elements are constructed from one data element each.
// CHECK-LABEL: define {{.*}}void @_Z3f12v(
// CHECK: %[[A12:.*]] = call {{.*}}ptr @_Znam(i64 noundef 2)
// CHECK: call void @_ZN1AC1Ec(ptr {{.*}}%[[A12]], i8 noundef signext 48)
// CHECK: %[[F12E1:.*]] = getelementptr inbounds %struct.A, ptr %[[A12]], i64 1
// CHECK: call void @_ZN1AC1Ec(ptr {{.*}}%[[F12E1]], i8 noundef signext 49)
void f12() {
  A *p = new A[]{
#embed <embed-data.txt> limit(2)
  };
}
