// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// We declare anonymous record types to represent lambdas. Rather than trying to
// to match the declarations, we establish variables for these when they are used.

int g3() {
  auto* fn = +[](int const& i) -> int { return i; };
  auto task = fn(3);
  return task;
}

// The order of these functions is different in OGCG.

// OGCG: define dso_local noundef i32 @_Z2g3v()
// OGCG:   %[[FN_PTR:.*]] = alloca ptr
// OGCG:   %[[REF_TMP:.*]] = alloca %[[REC_LAM_G3:.*]]
// OGCG:   %[[TASK:.*]] = alloca i32
// OGCG:   %[[REF_TMP1:.*]] = alloca i32
// OGCG:   %[[CALL:.*]] = call {{.*}} ptr @"_ZZ2g3vENK3$_0cvPFiRKiEEv"(ptr {{.*}} %[[REF_TMP]])
// OGCG:   store ptr %[[CALL]], ptr %[[FN_PTR]]
// OGCG:   %[[FN:.*]] = load ptr, ptr %[[FN_PTR]]
// OGCG:   store i32 3, ptr %[[REF_TMP1]]
// OGCG:   %[[CALL2:.*]] = call {{.*}} i32 %[[FN]](ptr {{.*}} %[[REF_TMP1]])
// OGCG:   store i32 %[[CALL2]], ptr %[[TASK]]
// OGCG:   %[[RESULT:.*]] = load i32, ptr %[[TASK]]
// OGCG:   ret i32 %[[RESULT]]

// OGCG: define internal noundef ptr @"_ZZ2g3vENK3$_0cvPFiRKiEEv"(ptr {{.*}} %[[THIS_ARG:.*]])
// OGCG:   %[[THIS_ADDR:.*]] = alloca ptr
// OGCG:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// OGCG:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// OGCG:   ret ptr @"_ZZ2g3vEN3$_08__invokeERKi"

// lambda operator()
// CIR: cir.func lambda internal private dso_local @_ZZ2g3vENK3$_0clERKi(%[[THIS_ARG:.*]]: !cir.ptr<![[REC_LAM_G3:.*]]> {{.*}}, %[[REF_I_ARG:.*]]: !cir.ptr<!s32i> {{.*}})
// CIR:   %[[THIS_ALLOCA:.*]] = cir.alloca !cir.ptr<![[REC_LAM_G3]]>, !cir.ptr<!cir.ptr<![[REC_LAM_G3]]>>, ["this", init]
// CIR:   %[[REF_I_ALLOCA:.*]] = cir.alloca {{.*}} ["i", init, const]
// CIR:   %[[RETVAL:.*]] = cir.alloca {{.*}} ["__retval"]
// CIR:   cir.store %[[THIS_ARG]], %[[THIS_ALLOCA]]
// CIR:   cir.store %[[REF_I_ARG]], %[[REF_I_ALLOCA]]
// CIR:   %[[THIS:.*]] = cir.load %[[THIS_ALLOCA]]
// CIR:   %[[REF_I:.*]] = cir.load %[[REF_I_ALLOCA]]
// CIR:   %[[I:.*]] = cir.load{{.*}} %[[REF_I]]
// CIR:   cir.store %[[I]], %[[RETVAL]]
// CIR:   %[[RET:.*]] = cir.load %[[RETVAL]]
// CIR:   cir.return %[[RET]]

// LLVM: define internal i32 @"_ZZ2g3vENK3$_0clERKi"(ptr %[[THIS_ARG:.*]], ptr %[[REF_I_ARG:.*]]){{.*}} {
// LLVM:   %[[THIS_ALLOCA:.*]] = alloca ptr
// LLVM:   %[[REF_I_ALLOCA:.*]] = alloca ptr
// LLVM:   %[[RETVAL:.*]] = alloca i32
// LLVM:   store ptr %[[THIS_ARG]], ptr %[[THIS_ALLOCA]]
// LLVM:   store ptr %[[REF_I_ARG]], ptr %[[REF_I_ALLOCA]]
// LLVM:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ALLOCA]]
// LLVM:   %[[REF_I:.*]] = load ptr, ptr %[[REF_I_ALLOCA]]
// LLVM:   %[[I:.*]] = load i32, ptr %[[REF_I]]
// LLVM:   store i32 %[[I]], ptr %[[RETVAL]]
// LLVM:   %[[RET:.*]] = load i32, ptr %[[RETVAL]]
// LLVM:   ret i32 %[[RET]]

// In OGCG, the _ZZ2g3vENK3$_0clERKi function is emitted after _ZZ2g3vEN3$_08__invokeERKi, see below.

// lambda invoker
// CIR: cir.func internal private dso_local @_ZZ2g3vEN3$_08__invokeERKi(%[[REF_I_ARG:.*]]: !cir.ptr<!s32i> {{.*}}) -> !s32i{{.*}} {
// CIR:   %[[REF_I_ALLOCA:.*]] = cir.alloca {{.*}} ["i", init, const]
// CIR:   %[[RETVAL:.*]] = cir.alloca {{.*}} ["__retval"]
// CIR:   %[[LAM_ALLOCA:.*]] = cir.alloca ![[REC_LAM_G3]], !cir.ptr<![[REC_LAM_G3]]>, ["unused.capture"]
// CIR:   cir.store %[[REF_I_ARG]], %[[REF_I_ALLOCA]]
// CIR:   %[[REF_I:.*]] = cir.load{{.*}} %[[REF_I_ALLOCA]]
// CIR:   %[[LAM_RESULT:.*]] = cir.call @_ZZ2g3vENK3$_0clERKi(%2, %3) : (!cir.ptr<![[REC_LAM_G3]]>, !cir.ptr<!s32i>) -> !s32i
// CIR:   cir.store{{.*}} %[[LAM_RESULT]], %[[RETVAL]]
// CIR:   %[[RET:.*]] = cir.load %[[RETVAL]]
// CIR:   cir.return %[[RET]]

// LLVM: define internal i32 @"_ZZ2g3vEN3$_08__invokeERKi"(ptr %[[REF_I_ARG:.*]]){{.*}} {
// LLVM:   %[[REF_I_ALLOCA:.*]] = alloca ptr
// LLVM:   %[[RETVAL:.*]] = alloca i32
// LLVM:   %[[LAM_ALLOCA:.*]] = alloca %[[REC_LAM_G3:.*]],
// LLVM:   store ptr %[[REF_I_ARG]], ptr %[[REF_I_ALLOCA]]
// LLVM:   %[[REF_I:.*]] = load ptr, ptr %[[REF_I_ALLOCA]]
// LLVM:   %[[LAM_RESULT:.*]] = call i32 @"_ZZ2g3vENK3$_0clERKi"(ptr %[[LAM_ALLOCA]], ptr %[[REF_I]])
// LLVM:   store i32 %[[LAM_RESULT]], ptr %[[RETVAL]]
// LLVM:   %[[RET:.*]] = load i32, ptr %[[RETVAL]]
// LLVM:   ret i32 %[[RET]]

// In OGCG, the _ZZ2g3vEN3$_08__invokeERKi function is emitted after _ZN1A3barEv, see below.

// lambda operator int (*)(int const&)()
// CIR:   cir.func internal private dso_local @_ZZ2g3vENK3$_0cvPFiRKiEEv(%[[THIS_ARG:.*]]: !cir.ptr<![[REC_LAM_G3]]> {{.*}}) -> !cir.ptr<!cir.func<(!cir.ptr<!s32i>) -> !s32i>>{{.*}} {
// CIR:   %[[THIS_ALLOCA:.*]] = cir.alloca !cir.ptr<![[REC_LAM_G3]]>, !cir.ptr<!cir.ptr<![[REC_LAM_G3]]>>, ["this", init]
// CIR:   %[[RETVAL:.*]] = cir.alloca !cir.ptr<!cir.func<(!cir.ptr<!s32i>) -> !s32i>>, !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!s32i>) -> !s32i>>>, ["__retval"]
// CIR:   cir.store %[[THIS_ARG]], %[[THIS_ALLOCA]]
// CIR:   %[[THIS:.*]] = cir.load %[[THIS_ALLOCA]]
// CIR:   %[[INVOKER:.*]] = cir.get_global @_ZZ2g3vEN3$_08__invokeERKi : !cir.ptr<!cir.func<(!cir.ptr<!s32i>) -> !s32i>>
// CIR:   cir.store %[[INVOKER]], %[[RETVAL]]
// CIR:   %[[RET:.*]] = cir.load %[[RETVAL]]
// CIR:   cir.return %[[RET]]

// LLVM: define internal ptr @"_ZZ2g3vENK3$_0cvPFiRKiEEv"(ptr %[[THIS_ARG:.*]]){{.*}} {
// LLVM:  %[[THIS_ALLOCA:.*]] = alloca ptr
// LLVM:  %[[RETVAL:.*]] = alloca ptr
// LLVM:  store ptr %[[THIS_ARG]], ptr %[[THIS_ALLOCA]]
// LLVM:  %[[THIS:.*]] = load ptr, ptr %[[THIS_ALLOCA]]
// LLVM:  store ptr @"_ZZ2g3vEN3$_08__invokeERKi", ptr %[[RETVAL]]
// LLVM:  %[[RET:.*]] = load ptr, ptr %[[RETVAL]]
// LLVM:  ret ptr %[[RET]]

// In OGCG, the _ZZ2g3vENK3$_0cvPFiRKiEEv function is emitted just after the _Z2g3v function, see above.

// CIR: cir.func{{.*}} @_Z2g3v() -> !s32i{{.*}} {
// CIR:   %[[RETVAL:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR:   %[[FN_ADDR:.*]] = cir.alloca !cir.ptr<!cir.func<(!cir.ptr<!s32i>) -> !s32i>>, !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!s32i>) -> !s32i>>>, ["fn", init]
// CIR:   %[[TASK:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["task", init]

// 1. Use `operator int (*)(int const&)()` to retrieve the fnptr to `__invoke()`.
// CIR:     %[[SCOPE_RET:.*]] = cir.scope {
// CIR:       %[[LAM_ALLOCA:.*]] = cir.alloca ![[REC_LAM_G3]], !cir.ptr<![[REC_LAM_G3]]>, ["ref.tmp0"]
// CIR:       %[[OPERATOR_RESULT:.*]] = cir.call @_ZZ2g3vENK3$_0cvPFiRKiEEv(%[[LAM_ALLOCA]]){{.*}}
// CIR:       %[[PLUS:.*]] = cir.unary(plus, %[[OPERATOR_RESULT]])
// CIR:       cir.yield %[[PLUS]]
// CIR:     }

// 2. Load ptr to `__invoke()`.
// CIR:     cir.store{{.*}} %[[SCOPE_RET]], %[[FN_ADDR]]
// CIR:     %[[SCOPE_RET2:.*]] = cir.scope {
// CIR:       %[[REF_TMP1:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["ref.tmp1", init]
// CIR:       %[[FN:.*]] = cir.load{{.*}} %[[FN_ADDR]]
// CIR:       %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
// CIR:       cir.store{{.*}} %[[THREE]], %[[REF_TMP1]]

// 3. Call `__invoke()`, which effectively executes `operator()`.
// CIR:       %[[RESULT:.*]] = cir.call %[[FN]](%[[REF_TMP1]])
// CIR:       cir.yield %[[RESULT]]
// CIR:     }

// CIR:     cir.store{{.*}} %[[SCOPE_RET2]], %[[TASK]]
// CIR:     %[[TASK_RET:.*]] = cir.load{{.*}} %[[TASK]]
// CIR:     cir.store{{.*}} %[[TASK_RET]], %[[RETVAL]]
// CIR:     %[[RET:.*]] = cir.load{{.*}} %[[RETVAL]]
// CIR:     cir.return %[[RET]]
// CIR:   }

// LLVM: define dso_local i32 @_Z2g3v(){{.*}} {
// LLVM:   %[[LAM_ALLOCA:.*]] = alloca %[[REC_LAM_G3]]
// LLVM:   %[[REF_TMP1:.*]] = alloca i32
// LLVM:   %[[RETVAL:.*]] = alloca i32
// LLVM:   %[[FN_PTR:.*]] = alloca ptr
// LLVM:   %[[TASK:.*]] = alloca i32
// LLVM:   br label %[[SCOPE_BB0:.*]]

// LLVM: [[SCOPE_BB0]]:
// LLVM:   %[[OPERATOR_RESULT:.*]] = call ptr @"_ZZ2g3vENK3$_0cvPFiRKiEEv"(ptr %[[LAM_ALLOCA]])
// LLVM:   br label %[[SCOPE_BB1:.*]]

// LLVM: [[SCOPE_BB1]]:
// LLVM:   %[[TMP0:.*]] = phi ptr [ %[[OPERATOR_RESULT]], %[[SCOPE_BB0]] ]
// LLVM:   store ptr %[[TMP0]], ptr %[[FN_PTR]]
// LLVM:   br label %[[SCOPE_BB2:.*]]

// LLVM: [[SCOPE_BB2]]:
// LLVM:   %[[FN:.*]] = load ptr, ptr %[[FN_PTR]]
// LLVM:   store i32 3, ptr %[[REF_TMP1]]
// LLVM:   %[[RESULT:.*]] = call i32 %[[FN]](ptr %[[REF_TMP1]])
// LLVM:   br label %[[RET_BB:.*]]

// LLVM: [[RET_BB]]:
// LLVM:   %[[TMP1:.*]] = phi i32 [ %[[RESULT]], %[[SCOPE_BB2]] ]
// LLVM:   store i32 %[[TMP1]], ptr %[[TASK]]
// LLVM:   %[[TMP2:.*]] = load i32, ptr %[[TASK]]
// LLVM:   store i32 %[[TMP2]], ptr %[[RETVAL]]
// LLVM:   %[[RET:.*]] = load i32, ptr %[[RETVAL]]
// LLVM:   ret i32 %[[RET]]

// The definition for _Z2g3v in OGCG is first among the functions for the g3 test, see above.

// The functions below are emitted later in OGCG, see above for the corresponding LLVM checks.

// OGCG: define internal noundef i32 @"_ZZ2g3vEN3$_08__invokeERKi"(ptr {{.*}} %[[I_ARG:.*]])
// OGCG:   %[[I_ADDR:.*]] = alloca ptr
// OGCG:   %[[UNUSED_CAPTURE:.*]] = alloca %[[REC_LAM_G3:.*]]
// OGCG:   store ptr %[[I_ARG]], ptr %[[I_ADDR]]
// OGCG:   %[[I_PTR:.*]] = load ptr, ptr %[[I_ADDR]]
// OGCG:   %[[CALL:.*]] = call {{.*}} i32 @"_ZZ2g3vENK3$_0clERKi"(ptr {{.*}} %[[UNUSED_CAPTURE]], ptr {{.*}} %[[I_PTR]])
// OGCG:   ret i32 %[[CALL]]

// OGCG: define internal noundef i32 @"_ZZ2g3vENK3$_0clERKi"(ptr {{.*}} %[[THIS_ARG:.*]], ptr {{.*}} %[[I_ARG:.*]])
// OGCG:   %[[THIS_ADDR:.*]] = alloca ptr
// OGCG:   %[[I_ADDR:.*]] = alloca ptr
// OGCG:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// OGCG:   store ptr %[[I_ARG]], ptr %[[I_ADDR]]
// OGCG:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// OGCG:   %[[I_PTR:.*]] = load ptr, ptr %[[I_ADDR]]
// OGCG:   %[[I:.*]] = load i32, ptr %[[I_PTR]]
// OGCG:   ret i32 %[[I]]
