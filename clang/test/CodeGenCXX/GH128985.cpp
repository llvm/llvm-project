// RUN: %clang_cc1 %s -triple x86_64 -emit-llvm -o - | FileCheck %s

// GH128985: #embed in the braced initializer of an array new-expression
// asserted in codegen.
// The first four bytes of this file are '/', '/', ' ', 'R' (47, 47, 32, 82).

// CHECK-LABEL: define {{.*}}void @_Z2f1i(
// CHECK: icmp ult i64 %{{.*}}, 4
// CHECK: %[[A1:.*]] = call {{.*}}ptr @_Znam(i64 {{.*}})
// CHECK: store i32 47, ptr %[[A1]]
// CHECK: %[[F1E1:.*]] = getelementptr inbounds i32, ptr %[[A1]], i64 1
// CHECK: store i32 47, ptr %[[F1E1]]
// CHECK: %[[F1E2:.*]] = getelementptr inbounds i32, ptr %[[F1E1]], i64 1
// CHECK: store i32 32, ptr %[[F1E2]]
// CHECK: %[[F1E3:.*]] = getelementptr inbounds i32, ptr %[[F1E2]], i64 1
// CHECK: store i32 82, ptr %[[F1E3]]
// CHECK: %[[F1REST:.*]] = sub i64 %{{.*}}, 16
// CHECK: call void @llvm.memset.p0.i64(ptr align 4 %{{.*}}, i8 0, i64 %[[F1REST]], i1 false)
void f1(int x) {
  int *p = new int[x]{
#embed __FILE__ limit(4)
  };
}

// CHECK-LABEL: define {{.*}}void @_Z2f2i(
// CHECK: icmp ult i64 %{{.*}}, 4
// CHECK: %[[A2:.*]] = call {{.*}}ptr @_Znam(i64 {{.*}})
// CHECK: store i32 500, ptr %[[A2]]
// CHECK: %[[F2E1:.*]] = getelementptr inbounds i32, ptr %[[A2]], i64 1
// CHECK: store i32 47, ptr %[[F2E1]]
// CHECK: %[[F2E2:.*]] = getelementptr inbounds i32, ptr %[[F2E1]], i64 1
// CHECK: store i32 47, ptr %[[F2E2]]
// CHECK: %[[F2E3:.*]] = getelementptr inbounds i32, ptr %[[F2E2]], i64 1
// CHECK: store i32 600, ptr %[[F2E3]]
// CHECK: %[[F2REST:.*]] = sub i64 %{{.*}}, 16
// CHECK: call void @llvm.memset.p0.i64(ptr align 4 %{{.*}}, i8 0, i64 %[[F2REST]], i1 false)
void f2(int x) {
  int *p = new int[x]{
    500,
#embed __FILE__ limit(2) suffix(, 600)
  };
}

// char arrays are initialized from the embed data via the string literal
// initialization path.
// CHECK-LABEL: define {{.*}}void @_Z2f3i(
// CHECK: icmp ult i64 %{{.*}}, 4
// CHECK: %[[A3:.*]] = call {{.*}}ptr @_Znam(i64 {{.*}})
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 1 %[[A3]], ptr align 1 @{{.*}}, i64 4, i1 false)
// CHECK: %[[F3END:.*]] = getelementptr inbounds i8, ptr %[[A3]], i64 4
// CHECK: %[[F3REST:.*]] = sub i64 %{{.*}}, 4
// CHECK: call void @llvm.memset.p0.i64(ptr align 1 %[[F3END]], i8 0, i64 %[[F3REST]], i1 false)
void f3(int x) {
  char *p = new char[x]{
#embed __FILE__ limit(4)
  };
}

// CHECK-LABEL: define {{.*}}void @_Z2f4i(
// CHECK: icmp ult i64 %{{.*}}, 2
// CHECK: %[[A4:.*]] = call {{.*}}ptr @_Znam(i64 {{.*}})
// CHECK: store i32 900, ptr %[[A4]]
// CHECK: %[[F4E1:.*]] = getelementptr inbounds i32, ptr %[[A4]], i64 1
// CHECK: store i32 47, ptr %[[F4E1]]
// CHECK: %[[F4REST:.*]] = sub i64 %{{.*}}, 8
// CHECK: call void @llvm.memset.p0.i64(ptr align 4 %{{.*}}, i8 0, i64 %[[F4REST]], i1 false)
void f4(int x) {
  int *p = new int[x]{
#embed __FILE__ limit(1) prefix(900, )
  };
}

// Constant size fully covered by the embed data: no trailing fill.
// CHECK-LABEL: define {{.*}}void @_Z2f5v(
// CHECK: %[[A5:.*]] = call {{.*}}ptr @_Znam(i64 {{.*}})
// CHECK: store i32 47, ptr %[[A5]]
// CHECK: %[[F5E1:.*]] = getelementptr inbounds i32, ptr %[[A5]], i64 1
// CHECK: store i32 47, ptr %[[F5E1]]
// CHECK: %[[F5E2:.*]] = getelementptr inbounds i32, ptr %[[F5E1]], i64 1
// CHECK: store i32 32, ptr %[[F5E2]]
// CHECK: %[[F5E3:.*]] = getelementptr inbounds i32, ptr %[[F5E2]], i64 1
// CHECK: store i32 82, ptr %[[F5E3]]
// CHECK-NOT: call void @llvm.memset
// CHECK: ret void
void f5() {
  int *p = new int[4]{
#embed __FILE__ limit(4)
  };
}
