// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

struct Delegating {
  Delegating();
  Delegating(int);
};

// Check that the constructor being delegated to is called with the correct
// arguments.
Delegating::Delegating() : Delegating(0) {}

// CIR: cir.func {{.*}} @_ZN10DelegatingC2Ev(%[[THIS_ARG:.*]]: !cir.ptr<!rec_Delegating> {{.*}})
// CIR:   %[[THIS_ADDR:.*]] = cir.alloca !cir.ptr<!rec_Delegating>, !cir.ptr<!cir.ptr<!rec_Delegating>>, ["this", init]
// CIR:   cir.store{{.*}} %[[THIS_ARG]], %[[THIS_ADDR]]
// CIR:   %[[THIS:.*]] = cir.load %[[THIS_ADDR]]
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   cir.call @_ZN10DelegatingC2Ei(%[[THIS]], %[[ZERO]]) : (!cir.ptr<!rec_Delegating>, !s32i) -> ()

// LLVM: define {{.*}} @_ZN10DelegatingC2Ev(ptr %[[THIS_ARG:.*]])
// LLVM:   %[[THIS_ADDR:.*]] = alloca ptr
// LLVM:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// LLVM:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// LLVM:   call void @_ZN10DelegatingC2Ei(ptr %[[THIS]], i32 0)

// OGCG: define {{.*}} @_ZN10DelegatingC2Ev(ptr {{.*}} %[[THIS_ARG:.*]])
// OGCG:   %[[THIS_ADDR:.*]] = alloca ptr
// OGCG:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// OGCG:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// OGCG:   call void @_ZN10DelegatingC2Ei(ptr {{.*}} %[[THIS]], i32 {{.*}} 0)

struct DelegatingWithZeroing {
  int i;
  DelegatingWithZeroing() = default;
  DelegatingWithZeroing(int);
};

// Check that the delegating constructor performs zero-initialization here.
// FIXME: we should either emit the trivial default constructor or remove the
// call to it in a lowering pass.
DelegatingWithZeroing::DelegatingWithZeroing(int) : DelegatingWithZeroing() {}

// CIR: cir.func {{.*}} @_ZN21DelegatingWithZeroingC2Ei(%[[THIS_ARG:.*]]: !cir.ptr<!rec_DelegatingWithZeroing> {{.*}}, %[[I_ARG:.*]]: !s32i {{.*}})
// CIR:   %[[THIS_ADDR:.*]] = cir.alloca !cir.ptr<!rec_DelegatingWithZeroing>, !cir.ptr<!cir.ptr<!rec_DelegatingWithZeroing>>, ["this", init]
// CIR:   %[[I_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["", init]
// CIR:   cir.store{{.*}} %[[THIS_ARG]], %[[THIS_ADDR]]
// CIR:   cir.store{{.*}} %[[I_ARG]], %[[I_ADDR]]
// CIR:   %[[THIS:.*]] = cir.load %[[THIS_ADDR]]
// CIR:   %[[ZERO:.*]] = cir.const #cir.zero : !rec_DelegatingWithZeroing
// CIR:   cir.store{{.*}} %[[ZERO]], %[[THIS]] : !rec_DelegatingWithZeroing, !cir.ptr<!rec_DelegatingWithZeroing>

// LLVM: define {{.*}} void @_ZN21DelegatingWithZeroingC2Ei(ptr %[[THIS_ARG:.*]], i32 %[[I_ARG:.*]])
// LLVM:   %[[THIS_ADDR:.*]] = alloca ptr
// LLVM:   %[[I_ADDR:.*]] = alloca i32
// LLVM:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// LLVM:   store i32 %[[I_ARG]], ptr %[[I_ADDR]]
// LLVM:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// LLVM:   store %struct.DelegatingWithZeroing zeroinitializer, ptr %[[THIS]]

// Note: OGCG elides the call to the default constructor.

// OGCG: define {{.*}} void @_ZN21DelegatingWithZeroingC2Ei(ptr {{.*}} %[[THIS_ARG:.*]], i32 {{.*}} %[[I_ARG:.*]])
// OGCG:   %[[THIS_ADDR:.*]] = alloca ptr
// OGCG:   %[[I_ADDR:.*]] = alloca i32
// OGCG:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// OGCG:   store i32 %[[I_ARG]], ptr %[[I_ADDR]]
// OGCG:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// OGCG:   call void @llvm.memset.p0.i64(ptr align 4 %[[THIS]], i8 0, i64 4, i1 false)
