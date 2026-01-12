// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c89 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c89 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c89 -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

// CIR: cir.func {{.*}}@foo(%arg0: !s32i
// CIR:   %0 = cir.alloca !s16i, !cir.ptr<!s16i>, ["x", init]
// CIR:   %1 = cir.cast integral %arg0 : !s32i -> !s16i
// CIR:   cir.store %1, %0 : !s16i, !cir.ptr<!s16i>
// expected-warning@+1 {{a function definition without a prototype is deprecated}}
void foo(x) short x; {}

// LLVM: define{{.*}} void @foo(i32 %0)
// LLVM:   %[[X_PTR:.*]] = alloca i16, i64 1, align 2
// LLVM:   %[[X:.*]] = trunc i32 %0 to i16
// LLVM:   store i16 %[[X]], ptr %[[X_PTR]], align 2

// OGCG: define{{.*}} void @foo(i32 noundef %0)
// OGCG: entry:
// OGCG:   %[[X_PTR:.*]] = alloca i16, align 2
// OGCG:   %[[X:.*]] = trunc i32 %0 to i16
// OGCG:   store i16 %[[X]], ptr %[[X_PTR]], align 2

// CIR: cir.func{{.*}}no_proto dso_local @bar(%arg0: !cir.double
// CIR:   %0 = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["f", init]
// CIR:   %1 = cir.cast floating %arg0 : !cir.double -> !cir.float
// CIR:   cir.store %1, %0 : !cir.float, !cir.ptr<!cir.float>
// expected-warning@+1 {{a function definition without a prototype is deprecated}}
void bar(f) float f; {}

// LLVM: define{{.*}} void @bar(double %0)
// LLVM:   %[[F_PTR:.*]] = alloca float, i64 1, align 4
// LLVM:   %[[F:.*]] = fptrunc double %0 to float
// LLVM:   store float %[[F]], ptr %[[F_PTR]], align 4

// OGCG: define{{.*}} void @bar(double noundef %0)
// OGCG: entry:
// OGCG:   %[[F_PTR:.*]] = alloca float, align 4
// OGCG:   %[[F:.*]] = fptrunc double %0 to float
// OGCG:   store float %[[F]], ptr %[[F_PTR]], align 4
