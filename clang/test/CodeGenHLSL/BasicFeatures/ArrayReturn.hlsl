// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -disable-llvm-passes -emit-llvm -finclude-default-header -o - %s | FileCheck %s

typedef int Foo[2];

// CHECK-LABEL: define void {{.*}}boop{{.*}}(ptr dead_on_unwind noalias writable sret([2 x i32]) align 4 %agg.result)
// CHECK: [[G:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[G]], ptr align 4 {{.*}}, i32 8, i1 false)
// CHECK-NEXT: [[AIB:%.*]] = getelementptr inbounds [2 x i32], ptr %agg.result, i32 0, i32 0
// CHECK-NEXT: br label %arrayinit.body
// CHECK: arrayinit.body:
// CHECK-NEXT: [[AII:%.*]] = phi i32 [ 0, %entry ], [ %arrayinit.next, %arrayinit.body ]
// CHECK-NEXT: [[X:%.*]] = getelementptr inbounds i32, ptr [[AIB]], i32 [[AII]]
// CHECK-NEXT: [[AI:%.*]] = getelementptr inbounds nuw [2 x i32], ptr [[G]], i32 0, i32 [[AII]]
// CHECK-NEXT: [[Y:%.*]] = load i32, ptr [[AI]], align 4
// CHECK-NEXT: store i32 [[Y]], ptr [[X]], align 4
// CHECK-NEXT: [[AIN:%.*]] = add nuw i32 [[AII]], 1
// CHECK-NEXT: [[AID:%.*]] = icmp eq i32 [[AIN]], 2
// CHECK-NEXT: br i1 [[AID]], label %arrayinit.end, label %arrayinit.body
// CHECK: arrayinit.end:           
// CHECK-NEXT: ret void
export Foo boop() {
  Foo G = {1,2};
  return G;
}

// CHECK-LABEL: define void {{.*}}foo{{.*}}(ptr dead_on_unwind noalias writable sret([2 x i32]) align 4 %agg.result)
// CHECK: store i32 1, ptr %agg.result, align 4
// CHECK-NEXT: [[E:%.*]] = getelementptr inbounds i32, ptr %agg.result, i32 1
// CHECK-NEXT: store i32 2, ptr [[E]], align 4
// CHECK-NEXT: ret void
export int foo()[2] {
  return {1,2};
}
