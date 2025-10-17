// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// We declare anonymous record types to represent lambdas. Rather than trying to
// to match the declarations, we establish variables for these when they are used.

void fn() {
  auto a = [](){};
  a();
}

// CIR: cir.func lambda internal private dso_local @_ZZ2fnvENK3$_0clEv(%[[THIS_ARG:.*]]: !cir.ptr<![[REC_LAM_FN_A:.*]]> {{.*}}) {{.*}} {
// CIR:   %[[THIS:.*]] = cir.alloca !cir.ptr<![[REC_LAM_FN_A]]>, !cir.ptr<!cir.ptr<![[REC_LAM_FN_A]]>>, ["this", init]
// CIR:   cir.store %[[THIS_ARG]], %[[THIS]]
// CIR:   cir.load %[[THIS]]
// CIR:   cir.return

// CIR: cir.func dso_local @_Z2fnv() {{.*}} {
// CIR:   %[[A:.*]] = cir.alloca ![[REC_LAM_FN_A]], !cir.ptr<![[REC_LAM_FN_A]]>, ["a"]
// CIR:   cir.call @_ZZ2fnvENK3$_0clEv(%[[A]])

// LLVM: define internal void @"_ZZ2fnvENK3$_0clEv"(ptr %[[THIS_ARG:.*]])
// LLVM:   %[[THIS_ADDR:.*]] = alloca ptr
// LLVM:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// LLVM:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// LLVM:   ret void

// FIXME: parameter attributes should be emitted
// LLVM: define {{.*}} void @_Z2fnv()
// LLVM:   [[A:%.*]] = alloca %[[REC_LAM_FN_A:.*]], i64 1, align 1
// LLVM:   call void @"_ZZ2fnvENK3$_0clEv"(ptr [[A]])
// LLVM:   ret void

// OGCG: define {{.*}} void @_Z2fnv()
// OGCG:   %[[A:.*]] = alloca %[[REC_LAM_FN_A:.*]]
// OGCG:   call void @"_ZZ2fnvENK3$_0clEv"(ptr {{.*}} %[[A]])
// OGCG:   ret void

// OGCG: define internal void @"_ZZ2fnvENK3$_0clEv"(ptr {{.*}} %[[THIS_ARG:.*]])
// OGCG:   %[[THIS_ADDR:.*]] = alloca ptr
// OGCG:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// OGCG:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// OGCG:   ret void

void l0() {
  int i;
  auto a = [&](){ i = i + 1; };
  a();
}

// CIR: cir.func lambda internal private dso_local @_ZZ2l0vENK3$_0clEv(%[[THIS_ARG:.*]]: !cir.ptr<![[REC_LAM_L0_A:.*]]> {{.*}}) {{.*}} {
// CIR:   %[[THIS_ADDR:.*]] = cir.alloca !cir.ptr<![[REC_LAM_L0_A]]>, !cir.ptr<!cir.ptr<![[REC_LAM_L0_A]]>>, ["this", init] {alignment = 8 : i64}
// CIR:   cir.store %[[THIS_ARG]], %[[THIS_ADDR]]
// CIR:   %[[THIS:.*]] = cir.load %[[THIS_ADDR]]
// CIR:   %[[I_ADDR_ADDR:.*]] = cir.get_member %[[THIS]][0] {name = "i"}
// CIR:   %[[I_ADDR:.*]] = cir.load %[[I_ADDR_ADDR]]
// CIR:   %[[I:.*]] = cir.load align(4) %[[I_ADDR]]
// CIR:   %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR:   %[[I_PLUS_ONE:.*]] = cir.binop(add, %[[I]], %[[ONE]]) nsw
// CIR:   %[[I_ADDR_ADDR:.*]] = cir.get_member %[[THIS]][0] {name = "i"}
// CIR:   %[[I_ADDR:.*]] = cir.load %[[I_ADDR_ADDR]]
// CIR:   cir.store{{.*}} %[[I_PLUS_ONE]], %[[I_ADDR]]
// CIR:   cir.return

// CIR: cir.func {{.*}} @_Z2l0v() {{.*}} {
// CIR:   %[[I:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i"]
// CIR:   %[[A:.*]] = cir.alloca ![[REC_LAM_L0_A]], !cir.ptr<![[REC_LAM_L0_A]]>, ["a", init]
// CIR:   %[[I_ADDR:.*]] = cir.get_member %[[A]][0] {name = "i"}
// CIR:   cir.store{{.*}} %[[I]], %[[I_ADDR]]
// CIR:   cir.call @_ZZ2l0vENK3$_0clEv(%[[A]])
// CIR:   cir.return

// LLVM: define internal void @"_ZZ2l0vENK3$_0clEv"(ptr %[[THIS_ARG:.*]])
// LLVM:   %[[THIS_ADDR:.*]] = alloca ptr
// LLVM:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// LLVM:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// LLVM:   %[[I_ADDR_ADDR:.*]] = getelementptr %[[REC_LAM_L0_A:.*]], ptr %[[THIS]], i32 0, i32 0
// LLVM:   %[[I_ADDR:.*]] = load ptr, ptr %[[I_ADDR_ADDR]]
// LLVM:   %[[I:.*]] = load i32, ptr %[[I_ADDR]]
// LLVM:   %[[ADD:.*]] = add nsw i32 %[[I]], 1
// LLVM:   %[[I_ADDR_ADDR:.*]] = getelementptr %[[REC_LAM_L0_A]], ptr %[[THIS]], i32 0, i32 0
// LLVM:   %[[I_ADDR:.*]] = load ptr, ptr %[[I_ADDR_ADDR]]
// LLVM:   store i32 %[[ADD]], ptr %[[I_ADDR]]
// LLVM:   ret void

// LLVM: define {{.*}} void @_Z2l0v()
// LLVM:   %[[I:.*]] = alloca i32
// LLVM:   %[[A:.*]] = alloca %[[REC_LAM_L0_A]]
// LLVM:   %[[I_ADDR:.*]] = getelementptr %[[REC_LAM_L0_A]], ptr %[[A]], i32 0, i32 0
// LLVM:   store ptr %[[I]], ptr %[[I_ADDR]]
// LLVM:   call void @"_ZZ2l0vENK3$_0clEv"(ptr %[[A]])
// LLVM:   ret void

// OGCG: define {{.*}} void @_Z2l0v()
// OGCG:   %[[I:.*]] = alloca i32
// OGCG:   %[[A:.*]] = alloca %[[REC_LAM_L0_A:.*]],
// OGCG:   %[[I_ADDR:.*]] = getelementptr inbounds nuw %[[REC_LAM_L0_A]], ptr %[[A]], i32 0, i32 0
// OGCG:   store ptr %[[I]], ptr %[[I_ADDR]]
// OGCG:   call void @"_ZZ2l0vENK3$_0clEv"(ptr {{.*}} %[[A]])
// OGCG:   ret void

// OGCG: define internal void @"_ZZ2l0vENK3$_0clEv"(ptr {{.*}} %[[THIS_ARG:.*]])
// OGCG:   %[[THIS_ADDR:.*]] = alloca ptr
// OGCG:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// OGCG:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// OGCG:   %[[I_ADDR_ADDR:.*]] = getelementptr inbounds nuw %[[REC_LAM_L0_A]], ptr %[[THIS]], i32 0, i32 0
// OGCG:   %[[I_ADDR:.*]] = load ptr, ptr %[[I_ADDR_ADDR]]
// OGCG:   %[[I:.*]] = load i32, ptr %[[I_ADDR]]
// OGCG:   %[[ADD:.*]] = add nsw i32 %[[I]], 1
// OGCG:   %[[I_ADDR_ADDR:.*]] = getelementptr inbounds nuw %[[REC_LAM_L0_A]], ptr %[[THIS]], i32 0, i32 0
// OGCG:   %[[I_ADDR:.*]] = load ptr, ptr %[[I_ADDR_ADDR]]
// OGCG:   store i32 %[[ADD]], ptr %[[I_ADDR]]
// OGCG:   ret void

auto g() {
  int i = 12;
  return [&] {
    i += 100;
    return i;
  };
}

// CIR: cir.func dso_local @_Z1gv() -> ![[REC_LAM_G:.*]] {{.*}} {
// CIR:   %[[RETVAL:.*]] = cir.alloca ![[REC_LAM_G]], !cir.ptr<![[REC_LAM_G]]>, ["__retval"]
// CIR:   %[[I_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init]
// CIR:   %[[TWELVE:.*]] = cir.const #cir.int<12> : !s32i
// CIR:   cir.store{{.*}} %[[TWELVE]], %[[I_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[I_ADDR_ADDR:.*]] = cir.get_member %[[RETVAL]][0] {name = "i"} : !cir.ptr<![[REC_LAM_G]]> -> !cir.ptr<!cir.ptr<!s32i>>
// CIR:   cir.store{{.*}} %[[I_ADDR]], %[[I_ADDR_ADDR]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR:   %[[RET:.*]] = cir.load{{.*}} %[[RETVAL]] : !cir.ptr<![[REC_LAM_G]]>, ![[REC_LAM_G]]
// CIR:   cir.return %[[RET]] : ![[REC_LAM_G]]

// Note: In this case, OGCG returns a pointer to the 'i' field of the lambda,
//       whereas CIR and LLVM return the lambda itself.

// LLVM: define dso_local %[[REC_LAM_G:.*]] @_Z1gv()
// LLVM:   %[[RETVAL:.*]] = alloca %[[REC_LAM_G]]
// LLVM:   %[[I:.*]] = alloca i32
// LLVM:   store i32 12, ptr %[[I]]
// LLVM:   %[[I_ADDR:.*]] = getelementptr %[[REC_LAM_G]], ptr %[[RETVAL]], i32 0, i32 0
// LLVM:   store ptr %[[I]], ptr %[[I_ADDR]]
// LLVM:   %[[RET:.*]] = load %[[REC_LAM_G]], ptr %[[RETVAL]]
// LLVM:   ret %[[REC_LAM_G]] %[[RET]]

// OGCG: define dso_local ptr @_Z1gv()
// OGCG:   %[[RETVAL:.*]] = alloca %[[REC_LAM_G:.*]],
// OGCG:   %[[I:.*]] = alloca i32
// OGCG:   store i32 12, ptr %[[I]]
// OGCG:   %[[I_ADDR:.*]] = getelementptr inbounds nuw %[[REC_LAM_G]], ptr %[[RETVAL]], i32 0, i32 0
// OGCG:   store ptr %[[I]], ptr %[[I_ADDR]]
// OGCG:   %[[COERCE_DIVE:.*]] = getelementptr inbounds nuw %[[REC_LAM_G]], ptr %[[RETVAL]], i32 0, i32 0
// OGCG:   %[[RET:.*]] = load ptr, ptr %[[COERCE_DIVE]]
// OGCG:   ret ptr %[[RET]]

auto g2() {
  int i = 12;
  auto lam = [&] {
    i += 100;
    return i;
  };
  return lam;
}

// Should be same as above because of NRVO
// CIR: cir.func dso_local @_Z2g2v() -> ![[REC_LAM_G2:.*]] {{.*}} {
// CIR:   %[[RETVAL:.*]] = cir.alloca ![[REC_LAM_G2]], !cir.ptr<![[REC_LAM_G2]]>, ["__retval", init]
// CIR:   %[[I_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init]
// CIR:   %[[TWELVE:.*]] = cir.const #cir.int<12> : !s32i
// CIR:   cir.store{{.*}} %[[TWELVE]], %[[I_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[I_ADDR_ADDR:.*]] = cir.get_member %[[RETVAL]][0] {name = "i"} : !cir.ptr<![[REC_LAM_G2]]> -> !cir.ptr<!cir.ptr<!s32i>>
// CIR:   cir.store{{.*}} %[[I_ADDR]], %[[I_ADDR_ADDR]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR:   %[[RET:.*]] = cir.load{{.*}} %[[RETVAL]] : !cir.ptr<![[REC_LAM_G2]]>, ![[REC_LAM_G2]]
// CIR:   cir.return %[[RET]] : ![[REC_LAM_G2]]

// LLVM: define dso_local %[[REC_LAM_G:.*]] @_Z2g2v()
// LLVM:   %[[RETVAL:.*]] = alloca %[[REC_LAM_G]]
// LLVM:   %[[I:.*]] = alloca i32
// LLVM:   store i32 12, ptr %[[I]]
// LLVM:   %[[I_ADDR:.*]] = getelementptr %[[REC_LAM_G]], ptr %[[RETVAL]], i32 0, i32 0
// LLVM:   store ptr %[[I]], ptr %[[I_ADDR]]
// LLVM:   %[[RET:.*]] = load %[[REC_LAM_G]], ptr %[[RETVAL]]
// LLVM:   ret %[[REC_LAM_G]] %[[RET]]

// OGCG: define dso_local ptr @_Z2g2v()
// OGCG:   %[[RETVAL:.*]] = alloca %[[REC_LAM_G2:.*]],
// OGCG:   %[[I:.*]] = alloca i32
// OGCG:   store i32 12, ptr %[[I]]
// OGCG:   %[[I_ADDR:.*]] = getelementptr inbounds nuw %[[REC_LAM_G2]], ptr %[[RETVAL]], i32 0, i32 0
// OGCG:   store ptr %[[I]], ptr %[[I_ADDR]]
// OGCG:   %[[COERCE_DIVE:.*]] = getelementptr inbounds nuw %[[REC_LAM_G2]], ptr %[[RETVAL]], i32 0, i32 0
// OGCG:   %[[RET:.*]] = load ptr, ptr %[[COERCE_DIVE]]
// OGCG:   ret ptr %[[RET]]

int f() {
  return g2()();
}

// CIR:cir.func lambda internal private dso_local @_ZZ2g2vENK3$_0clEv(%[[THIS_ARG:.*]]: !cir.ptr<![[REC_LAM_G2]]> {{.*}}) -> !s32i {{.*}} {
// CIR:   %[[THIS_ADDR:.*]] = cir.alloca !cir.ptr<![[REC_LAM_G2]]>, !cir.ptr<!cir.ptr<![[REC_LAM_G2]]>>, ["this", init]
// CIR:   %[[RETVAL:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR:   cir.store %[[THIS_ARG]], %[[THIS_ADDR]]
// CIR:   %[[THIS:.*]] = cir.load %[[THIS_ADDR]]
// CIR:   %[[ONE_HUNDRED:.*]] = cir.const #cir.int<100> : !s32i
// CIR:   %[[I_ADDR_ADDR:.*]] = cir.get_member %[[THIS]][0] {name = "i"}
// CIR:   %[[I_ADDR:.*]] = cir.load %[[I_ADDR_ADDR]]
// CIR:   %[[I:.*]] = cir.load{{.*}} %[[I_ADDR]]
// CIR:   %[[I_PLUS_ONE_HUNDRED:.*]] = cir.binop(add, %[[I]], %[[ONE_HUNDRED]]) nsw : !s32i
// CIR:   cir.store{{.*}} %[[I_PLUS_ONE_HUNDRED]], %[[I_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[I_ADDR_ADDR:.*]] = cir.get_member %[[THIS]][0] {name = "i"}
// CIR:   %[[I_ADDR:.*]] = cir.load %[[I_ADDR_ADDR]]
// CIR:   %[[I:.*]] = cir.load{{.*}} %[[I_ADDR]]
// CIR:   cir.store{{.*}} %[[I]], %[[RETVAL]]
// CIR:   %[[RET:.*]] = cir.load %[[RETVAL]]
// CIR:   cir.return %[[RET]]

// CIR: cir.func dso_local @_Z1fv() -> !s32i {{.*}} {
// CIR:   %[[RETVAL:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR:   %[[SCOPE_RET:.*]] = cir.scope {
// CIR:     %[[TMP:.*]] = cir.alloca ![[REC_LAM_G2]], !cir.ptr<![[REC_LAM_G2]]>, ["ref.tmp0"]
// CIR:     %[[G2:.*]] = cir.call @_Z2g2v() : () -> ![[REC_LAM_G2]]
// CIR:     cir.store{{.*}} %[[G2]], %[[TMP]]
// CIR:     %[[RESULT:.*]] = cir.call @_ZZ2g2vENK3$_0clEv(%[[TMP]])
// CIR:     cir.yield %[[RESULT]]
// CIR:   }
// CIR:   cir.store{{.*}} %[[SCOPE_RET]], %[[RETVAL]]
// CIR:   %[[RET:.*]] = cir.load{{.*}} %[[RETVAL]]
// CIR:   cir.return %[[RET]]

// LLVM: define internal i32 @"_ZZ2g2vENK3$_0clEv"(ptr %[[THIS_ARG:.*]])
// LLVM:   %[[THIS_ALLOCA:.*]] = alloca ptr
// LLVM:   %[[I_ALLOCA:.*]] = alloca i32
// LLVM:   store ptr %[[THIS_ARG]], ptr %[[THIS_ALLOCA]]
// LLVM:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ALLOCA]]
// LLVM:   %[[I_ADDR_ADDR:.*]] = getelementptr %[[REC_LAM_G2:.*]], ptr %[[THIS]], i32 0, i32 0
// LLVM:   %[[I_ADDR:.*]] = load ptr, ptr %[[I_ADDR_ADDR]]
// LLVM:   %[[I:.*]] = load i32, ptr %[[I_ADDR]]
// LLVM:   %[[ADD:.*]] = add nsw i32 %[[I]], 100
// LLVM:   store i32 %[[ADD]], ptr %[[I_ADDR]]
// LLVM:   %[[I_ADDR_ADDR:.*]] = getelementptr %[[REC_LAM_G2]], ptr %[[THIS]], i32 0, i32 0
// LLVM:   %[[I_ADDR:.*]] = load ptr, ptr %[[I_ADDR_ADDR]]
// LLVM:   %[[I:.*]] = load i32, ptr %[[I_ADDR]]
// LLVM:   store i32 %[[I]], ptr %[[I_ALLOCA]]
// LLVM:   %[[RET:.*]] = load i32, ptr %[[I_ALLOCA]]
// LLVM:   ret i32 %[[RET]]

// LLVM: define {{.*}} i32 @_Z1fv()
// LLVM:   %[[TMP:.*]] = alloca %[[REC_LAM_G2]]
// LLVM:   %[[RETVAL:.*]] = alloca i32
// LLVM:   br label %[[SCOPE_BB:.*]]
// LLVM: [[SCOPE_BB]]:
// LLVM:   %[[G2:.*]] = call %[[REC_LAM_G2]] @_Z2g2v()
// LLVM:   store %[[REC_LAM_G2]] %[[G2]], ptr %[[TMP]]
// LLVM:   %[[RESULT:.*]] = call i32 @"_ZZ2g2vENK3$_0clEv"(ptr %[[TMP]])
// LLVM:   br label %[[RET_BB:.*]]
// LLVM: [[RET_BB]]:
// LLVM:   %[[RETPHI:.*]] = phi i32 [ %[[RESULT]], %[[SCOPE_BB]] ]
// LLVM:   store i32 %[[RETPHI]], ptr %[[RETVAL]]
// LLVM:   %[[RET:.*]] = load i32, ptr %[[RETVAL]]
// LLVM:   ret i32 %[[RET]]

// The order of these functions is reversed in OGCG.

// OGCG: define {{.*}} i32 @_Z1fv()
// OGCG:   %[[TMP:.*]] = alloca %[[REC_LAM_G2]]
// OGCG:   %[[RESULT:.*]] = call ptr @_Z2g2v()
// OGCG:   %[[COERCE_DIVE:.*]] = getelementptr inbounds nuw %[[REC_LAM_G2]], ptr %[[TMP]], i32 0, i32 0
// OGCG:   store ptr %[[RESULT]], ptr %[[COERCE_DIVE]]
// OGCG:   %[[RET:.*]] = call {{.*}} i32 @"_ZZ2g2vENK3$_0clEv"(ptr {{.*}} %[[TMP]])
// OGCG:   ret i32 %[[RET]]

// OGCG: define internal noundef i32 @"_ZZ2g2vENK3$_0clEv"(ptr {{.*}} %[[THIS_ARG:.*]])
// OGCG:   %[[THIS_ALLOCA:.*]] = alloca ptr
// OGCG:   store ptr %[[THIS_ARG]], ptr %[[THIS_ALLOCA]]
// OGCG:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ALLOCA]]
// OGCG:   %[[I_ADDR_ADDR:.*]] = getelementptr inbounds nuw %[[REC_LAM_G2]], ptr %[[THIS]], i32 0, i32 0
// OGCG:   %[[I_ADDR:.*]] = load ptr, ptr %[[I_ADDR_ADDR]]
// OGCG:   %[[I:.*]] = load i32, ptr %[[I_ADDR]]
// OGCG:   %[[ADD:.*]] = add nsw i32 %[[I]], 100
// OGCG:   store i32 %[[ADD]], ptr %[[I_ADDR]]
// OGCG:   %[[I_ADDR_ADDR:.*]] = getelementptr inbounds nuw %[[REC_LAM_G2]], ptr %[[THIS]], i32 0, i32 0
// OGCG:   %[[I_ADDR:.*]] = load ptr, ptr %[[I_ADDR_ADDR]]
// OGCG:   %[[I:.*]] = load i32, ptr %[[I_ADDR]]
// OGCG:   ret i32 %[[I]]

struct A {
  int a = 111;
  int foo() { return [*this] { return a; }(); }
  int bar() { return [this] { return a; }(); }
};

// This function gets emitted before the lambdas in OGCG.

// OGCG: define {{.*}} i32 @_Z17test_lambda_this1v
// OGCG:   %[[A_THIS:.*]] = alloca %struct.A
// OGCG:   call void @_ZN1AC1Ev(ptr {{.*}} %[[A_THIS]])
// OGCG:   call noundef i32 @_ZN1A3fooEv(ptr {{.*}} %[[A_THIS]])
// OGCG:   call noundef i32 @_ZN1A3barEv(ptr {{.*}} %[[A_THIS]])

// lambda operator() in foo()
// CIR: cir.func lambda comdat linkonce_odr @_ZZN1A3fooEvENKUlvE_clEv(%[[THIS_ARG:.*]]: !cir.ptr<![[REC_LAM_A:.*]]> {{.*}}) {{.*}} {
// CIR:   %[[THIS_ADDR:.*]] = cir.alloca !cir.ptr<![[REC_LAM_A]]>, !cir.ptr<!cir.ptr<![[REC_LAM_A]]>>, ["this", init]
// CIR:   %[[RETVAL:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR:   cir.store{{.*}} %[[THIS_ARG]], %[[THIS_ADDR]]
// CIR:   %[[THIS:.*]] = cir.load{{.*}} %[[THIS_ADDR]]
// CIR:   %[[STRUCT_A:.*]] = cir.get_member %[[THIS]][0] {name = "this"}
// CIR:   %[[A_A_ADDR:.*]] = cir.get_member %[[STRUCT_A]][0] {name = "a"}
// CIR:   %[[A_A:.*]] = cir.load{{.*}} %[[A_A_ADDR]]
// CIR:   cir.store{{.*}} %[[A_A]], %[[RETVAL]]
// CIR:   %[[RET:.*]] = cir.load{{.*}} %[[RETVAL]]
// CIR:   cir.return %[[RET]]

// LLVM: define linkonce_odr i32 @_ZZN1A3fooEvENKUlvE_clEv(ptr %[[THIS_ARG:.*]])
// LLVM:   %[[THIS_ALLOCA:.*]]  = alloca ptr
// LLVM:   %[[RETVAL:.*]] = alloca i32
// LLVM:   store ptr %[[THIS_ARG]], ptr %[[THIS_ALLOCA]]
// LLVM:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ALLOCA]]
// LLVM:   %[[PTR_A:.*]] = getelementptr %[[REC_LAM_A:.*]], ptr %[[THIS]], i32 0, i32 0
// LLVM:   %[[A_A_ADDR:.*]] = getelementptr %struct.A, ptr %[[PTR_A]], i32 0, i32 0
// LLVM:   %[[A_A:.*]] = load i32, ptr %[[A_A_ADDR]]
// LLVM:   store i32 %[[A_A]], ptr %[[RETVAL]]
// LLVM:   %[[RET:.*]] = load i32, ptr %[[RETVAL]]
// LLVM:   ret i32 %[[RET]]

// The function above is defined after _ZN1A3barEv in OGCG, see below.

// A::foo()
// CIR: cir.func {{.*}} @_ZN1A3fooEv(%[[THIS_ARG:.*]]: !cir.ptr<!rec_A> {{.*}}) -> !s32i {{.*}} {
// CIR:   %[[THIS_ADDR:.*]] = cir.alloca !cir.ptr<!rec_A>, !cir.ptr<!cir.ptr<!rec_A>>, ["this", init]
// CIR:   %[[RETVAL:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR:   cir.store %[[THIS_ARG]], %[[THIS_ADDR]]
// CIR:   %[[THIS]] = cir.load deref %[[THIS_ADDR]] : !cir.ptr<!cir.ptr<!rec_A>>, !cir.ptr<!rec_A>
// CIR:   %[[SCOPE_RET:.*]] = cir.scope {
// CIR:     %[[LAM_ADDR:.*]] = cir.alloca ![[REC_LAM_A]], !cir.ptr<![[REC_LAM_A]]>, ["ref.tmp0"]
// CIR:     %[[STRUCT_A:.*]] = cir.get_member %[[LAM_ADDR]][0] {name = "this"} : !cir.ptr<![[REC_LAM_A]]> -> !cir.ptr<!rec_A>
// CIR:     cir.call @_ZN1AC1ERKS_(%[[STRUCT_A]], %[[THIS]]){{.*}} : (!cir.ptr<!rec_A>, !cir.ptr<!rec_A>){{.*}} -> ()
// CIR:     %[[LAM_RET:.*]] = cir.call @_ZZN1A3fooEvENKUlvE_clEv(%[[LAM_ADDR]])
// CIR:     cir.yield %[[LAM_RET]]
// CIR:   }
// CIR:   cir.store{{.*}} %[[SCOPE_RET]], %[[RETVAL]]
// CIR:   %[[RET:.*]] = cir.load{{.*}} %[[RETVAL]]
// CIR:   cir.return %[[RET]]

// LLVM: define linkonce_odr i32 @_ZN1A3fooEv(ptr %[[THIS_ARG:.*]])
// LLVM:   %[[LAM_ALLOCA:.*]] =  alloca %[[REC_LAM_A]]
// LLVM:   %[[THIS_ALLOCA:.*]] = alloca ptr
// LLVM:   %[[RETVAL:.*]] = alloca i32
// LLVM:   store ptr %[[THIS_ARG]], ptr %[[THIS_ALLOCA]]
// LLVM:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ALLOCA]]
// LLVM:   br label %[[SCOPE_BB:.*]]
// LLVM: [[SCOPE_BB]]:
// LLVM:   %[[STRUCT_A:.*]] = getelementptr %[[REC_LAM_A]], ptr %[[LAM_ALLOCA]], i32 0, i32 0
// LLVM:   call void @_ZN1AC1ERKS_(ptr %[[STRUCT_A]], ptr %[[THIS]])
// LLVM:   %[[LAM_RET:.*]] = call i32 @_ZZN1A3fooEvENKUlvE_clEv(ptr %[[LAM_ALLOCA]])
// LLVM:   br label %[[RET_BB:.*]]
// LLVM: [[RET_BB]]:
// LLVM:   %[[RETPHI:.*]] = phi i32 [ %[[LAM_RET]], %[[SCOPE_BB]] ]
// LLVM:   store i32 %[[RETPHI]], ptr %[[RETVAL]]
// LLVM:   %[[RET:.*]] = load i32, ptr %[[RETVAL]]
// LLVM:   ret i32 %[[RET]]

// OGCG: define linkonce_odr noundef i32 @_ZN1A3fooEv(ptr {{.*}} %[[THIS_ARG:.*]])
// OGCG:   %[[THIS_ALLOCA:.*]] = alloca ptr
// OGCG:   %[[LAM_ALLOCA:.*]] =  alloca %[[REC_LAM_A:.*]],
// OGCG:   store ptr %[[THIS_ARG]], ptr %[[THIS_ALLOCA]]
// OGCG:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ALLOCA]]
// OGCG:   %[[STRUCT_A:.*]] = getelementptr inbounds nuw %[[REC_LAM_A]], ptr %[[LAM_ALLOCA]], i32 0, i32 0
// OGCG:   call void @llvm.memcpy.p0.p0.i64(ptr {{.*}} %[[STRUCT_A]], ptr {{.*}} %[[THIS]], i64 4, i1 false)
// OGCG:   %[[LAM_RET:.*]] = call noundef i32 @_ZZN1A3fooEvENKUlvE_clEv(ptr {{.*}} %[[LAM_ALLOCA]])
// OGCG:   ret i32 %[[LAM_RET]]

// lambda operator() in bar()
// CIR: cir.func {{.*}} @_ZZN1A3barEvENKUlvE_clEv(%[[THIS_ARG2:.*]]: !cir.ptr<![[REC_LAM_PTR_A:.*]]> {{.*}}) -> !s32i {{.*}} {
// CIR:   %[[THIS_ADDR:.*]] = cir.alloca !cir.ptr<![[REC_LAM_PTR_A]]>, !cir.ptr<!cir.ptr<![[REC_LAM_PTR_A]]>>, ["this", init]
// CIR:   %[[RETVAL:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR:   cir.store{{.*}} %[[THIS_ARG]], %[[THIS_ADDR]]
// CIR:   %[[THIS:.*]] = cir.load{{.*}} %[[THIS_ADDR]]
// CIR:   %[[STRUCT_A_ADDR_ADDR:.*]] = cir.get_member %[[THIS]][0] {name = "this"}
// CIR:   %[[STRUCT_A_ADDR:.*]] = cir.load{{.*}} %[[STRUCT_A_ADDR_ADDR]]
// CIR:   %[[A_A_ADDR:.*]] = cir.get_member %[[STRUCT_A_ADDR]][0] {name = "a"}
// CIR:   %[[A_A:.*]] = cir.load{{.*}} %[[A_A_ADDR]]
// CIR:   cir.store{{.*}} %[[A_A]], %[[RETVAL]]
// CIR:   %[[RET:.*]] = cir.load{{.*}} %[[RETVAL]]
// CIR:   cir.return %[[RET]]

// LLVM: define linkonce_odr i32 @_ZZN1A3barEvENKUlvE_clEv(ptr %[[THIS_ARG:.*]])
// LLVM:   %[[THIS_ALLOCA:.*]]  = alloca ptr
// LLVM:   %[[RETVAL:.*]] = alloca i32
// LLVM:   store ptr %[[THIS_ARG]], ptr %[[THIS_ALLOCA]]
// LLVM:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ALLOCA]]
// LLVM:   %[[STRUCT_A_ADDRR_ADDR:.*]] = getelementptr %[[REC_LAM_PTR_A:.*]], ptr %[[THIS]], i32 0, i32 0
// LLVM:   %[[STRUCT_A_ADDR:.*]] = load ptr, ptr %[[STRUCT_A_ADDRR_ADDR]]
// LLVM:   %[[A_A_ADDR:.*]] = getelementptr %struct.A, ptr %[[STRUCT_A_ADDR]], i32 0, i32 0
// LLVM:   %[[A_A:.*]] = load i32, ptr %[[A_A_ADDR]]
// LLVM:   store i32 %[[A_A]], ptr %[[RETVAL]]
// LLVM:   %[[RET:.*]] = load i32, ptr %[[RETVAL]]
// LLVM:   ret i32 %[[RET]]

// The function above is defined after _ZZN1A3fooEvENKUlvE_clEv in OGCG, see below.

// A::bar()
// CIR: cir.func {{.*}} @_ZN1A3barEv(%[[THIS_ARG:.*]]: !cir.ptr<!rec_A> {{.*}}) -> !s32i {{.*}} {
// CIR:   %[[THIS_ADDR:.*]] = cir.alloca !cir.ptr<!rec_A>, !cir.ptr<!cir.ptr<!rec_A>>, ["this", init]
// CIR:   %[[RETVAL:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR:   cir.store %[[THIS_ARG]], %[[THIS_ADDR]]
// CIR:   %[[THIS]] = cir.load %[[THIS_ADDR]] : !cir.ptr<!cir.ptr<!rec_A>>, !cir.ptr<!rec_A>
// CIR:   %[[SCOPE_RET:.*]] = cir.scope {
// CIR:     %[[LAM_ADDR:.*]] = cir.alloca ![[REC_LAM_PTR_A]], !cir.ptr<![[REC_LAM_PTR_A]]>, ["ref.tmp0"]
// CIR:     %[[A_ADDR_ADDR:.*]] = cir.get_member %[[LAM_ADDR]][0] {name = "this"} : !cir.ptr<![[REC_LAM_PTR_A]]> -> !cir.ptr<!cir.ptr<!rec_A>>
// CIR:     cir.store{{.*}} %[[THIS]], %[[A_ADDR_ADDR]]
// CIR:     %[[LAM_RET:.*]] = cir.call @_ZZN1A3barEvENKUlvE_clEv(%[[LAM_ADDR]])
// CIR:     cir.yield %[[LAM_RET]]
// CIR:   }
// CIR:   cir.store{{.*}} %[[SCOPE_RET]], %[[RETVAL]]
// CIR:   %[[RET:.*]] = cir.load{{.*}} %[[RETVAL]]
// CIR:   cir.return %[[RET]]

// LLVM: define linkonce_odr i32 @_ZN1A3barEv(ptr %[[THIS_ARG:.*]])
// LLVM:   %[[LAM_ALLOCA:.*]] =  alloca %[[REC_LAM_PTR_A]]
// LLVM:   %[[THIS_ALLOCA:.*]] = alloca ptr
// LLVM:   %[[RETVAL:.*]] = alloca i32
// LLVM:   store ptr %[[THIS_ARG]], ptr %[[THIS_ALLOCA]]
// LLVM:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ALLOCA]]
// LLVM:   br label %[[SCOPE_BB:.*]]
// LLVM: [[SCOPE_BB]]:
// LLVM:   %[[A_ADDR_ADDR:.*]] = getelementptr %[[REC_LAM_PTR_A]], ptr %[[LAM_ALLOCA]], i32 0, i32 0
// LLVM:   store ptr %[[THIS]], ptr %[[A_ADDR_ADDR]]
// LLVM:   %[[LAM_RET:.*]] = call i32 @_ZZN1A3barEvENKUlvE_clEv(ptr %[[LAM_ALLOCA]])
// LLVM:   br label %[[RET_BB:.*]]
// LLVM: [[RET_BB]]:
// LLVM:   %[[RETPHI:.*]] = phi i32 [ %[[LAM_RET]], %[[SCOPE_BB]] ]
// LLVM:   store i32 %[[RETPHI]], ptr %[[RETVAL]]
// LLVM:   %[[RET:.*]] = load i32, ptr %[[RETVAL]]
// LLVM:   ret i32 %[[RET]]

// OGCG: define linkonce_odr noundef i32 @_ZN1A3barEv(ptr {{.*}} %[[THIS_ARG:.*]])
// OGCG:   %[[THIS_ALLOCA:.*]] = alloca ptr
// OGCG:   %[[LAM_ALLOCA:.*]] =  alloca %[[REC_LAM_PTR_A:.*]],
// OGCG:   store ptr %[[THIS_ARG]], ptr %[[THIS_ALLOCA]]
// OGCG:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ALLOCA]]
// OGCG:   %[[STRUCT_A:.*]] = getelementptr inbounds nuw %[[REC_LAM_PTR_A]], ptr %[[LAM_ALLOCA]], i32 0, i32 0
// OGCG:   store ptr %[[THIS]], ptr %[[STRUCT_A]]
// OGCG:   %[[LAM_RET:.*]] = call noundef i32 @_ZZN1A3barEvENKUlvE_clEv(ptr {{.*}} %[[LAM_ALLOCA]])
// OGCG:   ret i32 %[[LAM_RET]]

// OGCG: define linkonce_odr noundef i32 @_ZZN1A3fooEvENKUlvE_clEv(ptr {{.*}} %[[THIS_ARG:.*]])
// OGCG:   %[[THIS_ALLOCA:.*]]  = alloca ptr
// OGCG:   store ptr %[[THIS_ARG]], ptr %[[THIS_ALLOCA]]
// OGCG:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ALLOCA]]
// OGCG:   %[[PTR_A:.*]] = getelementptr inbounds nuw %[[REC_LAM_A]], ptr %[[THIS]], i32 0, i32 0
// OGCG:   %[[A_A_ADDR:.*]] = getelementptr inbounds nuw %struct.A, ptr %[[PTR_A]], i32 0, i32 0
// OGCG:   %[[A_A:.*]] = load i32, ptr %[[A_A_ADDR]]
// OGCG:   ret i32 %[[A_A]]

// OGCG: define linkonce_odr noundef i32 @_ZZN1A3barEvENKUlvE_clEv(ptr {{.*}} %[[THIS_ARG:.*]])
// OGCG:   %[[THIS_ALLOCA:.*]]  = alloca ptr
// OGCG:   store ptr %[[THIS_ARG]], ptr %[[THIS_ALLOCA]]
// OGCG:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ALLOCA]]
// OGCG:   %[[A_ADDR_ADDR:.*]] = getelementptr inbounds nuw %[[REC_LAM_PTR_A]], ptr %[[THIS]], i32 0, i32 0
// OGCG:   %[[A_ADDR:.*]] = load ptr, ptr %[[A_ADDR_ADDR]]
// OGCG:   %[[A_A_ADDR:.*]] = getelementptr inbounds nuw %struct.A, ptr %[[A_ADDR]], i32 0, i32 0
// OGCG:   %[[A_A:.*]] = load i32, ptr %[[A_A_ADDR]]
// OGCG:   ret i32 %[[A_A]]

int test_lambda_this1(){
  struct A clsA;
  int x = clsA.foo();
  int y = clsA.bar();
  return x+y;
}

// CIR: cir.func {{.*}} @_Z17test_lambda_this1v{{.*}} {
// CIR:   cir.call @_ZN1AC1Ev(%[[A_THIS:.*]]){{.*}} : (!cir.ptr<!rec_A>) -> ()
// CIR:   cir.call @_ZN1A3fooEv(%[[A_THIS]]){{.*}} : (!cir.ptr<!rec_A>) -> !s32i
// CIR:   cir.call @_ZN1A3barEv(%[[A_THIS]]){{.*}} : (!cir.ptr<!rec_A>) -> !s32i

// LLVM: define {{.*}} i32 @_Z17test_lambda_this1v
// LLVM:   %[[A_THIS:.*]] = alloca %struct.A
// LLVM:   call void @_ZN1AC1Ev(ptr %[[A_THIS]])
// LLVM:   call i32 @_ZN1A3fooEv(ptr %[[A_THIS]])
// LLVM:   call i32 @_ZN1A3barEv(ptr %[[A_THIS]])

// The function above is define before lambda operator() in foo() in OGCG, see above.
