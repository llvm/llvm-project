// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

static int a[10]{};
// CIR: cir.global "private" internal dso_local @_ZL1a = #cir.zero : !cir.array<!s32i x 10> {alignment = 16 : i64}
// LLVM: @_ZL1a = internal global [10 x i32] zeroinitializer, align 16

struct NonTrivialDestructor {
  ~NonTrivialDestructor();
};
struct BiggerNonTrivialDestructor {
  int array[12];
  ~BiggerNonTrivialDestructor();
};

void use() {
  for (int i : a) {}
  // This happens 3x (range + begin + end), but they all use the same code, sox
  // only test it 1x. Ensure the alignment is correct.
  // CIR: %[[GLOBAL_A:.*]] = cir.const #cir.global_view<@_ZL1a> : !cir.ptr<!cir.array<!s32i x 10>>
  // CIR: cir.store align(8) %[[GLOBAL_A]], %{{.*}} : !cir.ptr<!cir.array<!s32i x 10>>, !cir.ptr<!cir.ptr<!cir.array<!s32i x 10>>>
  // LLVM: store ptr getelementptr inbounds nuw (i8, ptr @_ZL1a, i64 40), ptr %{{.*}}, align 8

  // Make sure we get alignment correct here.
  NonTrivialDestructor a;
  // CIR-DAG: cir.call @_ZN20NonTrivialDestructorD1Ev{{.*}}llvm.align = 1
  // LLVM-DAG: call void @_ZN20NonTrivialDestructorD1Ev(ptr {{.*}}align 1
  BiggerNonTrivialDestructor b;
  // CIR-DAG: cir.call @_ZN26BiggerNonTrivialDestructorD1Ev{{.*}}llvm.align = 4
  // LLVM-DAG: call void @_ZN26BiggerNonTrivialDestructorD1Ev(ptr {{.*}}align 4
}
