// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -I%S/../Inputs -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -I%S/../Inputs -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -I%S/../Inputs -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

#include "std-cxx.h"

std::vector<const char*> test_nrvo() {
  std::vector<const char*> result;
  result.push_back("Words bend our thinking to infinite paths of self-delusion");
  return result;
}

// CIR: ![[VEC:.*]] = !cir.record<class "std::vector<const char *>" {!cir.ptr<!cir.ptr<!s8i>>, !cir.ptr<!cir.ptr<!s8i>>, !cir.ptr<!cir.ptr<!s8i>>}>

// CIR: cir.func {{.*}} @_Z9test_nrvov() -> ![[VEC]]
// CIR:   %[[RESULT:.*]] = cir.alloca ![[VEC]], !cir.ptr<![[VEC]]>, ["__retval", init]
// CIR:   %[[NRVO_FLAG:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["nrvo"]
// CIR:   %[[FALSE:.*]] = cir.const #false
// CIR:   cir.store{{.*}} %[[FALSE]], %[[NRVO_FLAG]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:   cir.call @_ZNSt6vectorIPKcEC1Ev(%[[RESULT]]) : (!cir.ptr<![[VEC]]>) -> ()
// CIR:   cir.scope {
// CIR:     %[[REF_TMP:.*]] = cir.alloca !cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>>, ["ref.tmp0"]
// CIR:     %[[STR:.*]] = cir.get_global @".str" : !cir.ptr<!cir.array<!s8i x 59>>
// CIR:     %[[PTR_DECAY:.*]] = cir.cast array_to_ptrdecay %[[STR]] : !cir.ptr<!cir.array<!s8i x 59>> -> !cir.ptr<!s8i>
// CIR:     cir.store{{.*}} %[[PTR_DECAY]], %[[REF_TMP]] : !cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>>
// CIR:     cir.call @_ZNSt6vectorIPKcE9push_backEOS1_(%[[RESULT]], %[[REF_TMP]]) : (!cir.ptr<![[VEC]]>, !cir.ptr<!cir.ptr<!s8i>>) -> ()
// CIR:   }
// CIR:   %[[TRUE:.*]] = cir.const #true
// CIR:   cir.store{{.*}} %[[TRUE]], %[[NRVO_FLAG]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:   %[[NRVO_FLAG_VAL:.*]] = cir.load{{.*}} %[[NRVO_FLAG]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR:   %[[NOT_NRVO:.*]] = cir.unary(not, %[[NRVO_FLAG_VAL]]) : !cir.bool, !cir.bool
// CIR:   cir.if %[[NOT_NRVO]] {
// CIR:     cir.call @_ZNSt6vectorIPKcED1Ev(%[[RESULT]]) : (!cir.ptr<!rec_std3A3Avector3Cconst_char_2A3E>) -> ()
// CIR:   }
// CIR:   %[[RETVAL:.*]] = cir.load{{.*}} %[[RESULT]] : !cir.ptr<![[VEC]]>, ![[VEC]]
// CIR:   cir.return %[[RETVAL]] : ![[VEC]]

// LLVM: define {{.*}} %[[VEC:.*]] @_Z9test_nrvov()
// LLVM:   %[[REF_TMP:.*]] = alloca ptr
// LLVM:   %[[RESULT:.*]] = alloca %[[VEC]]
// LLVM:   %[[NRVO_FLAG:.*]] = alloca i8
// LLVM:   store i8 0, ptr %[[NRVO_FLAG]]
// LLVM:   call void @_ZNSt6vectorIPKcEC1Ev(ptr %[[RESULT]])
// LLVM:   br label %[[SCOPE:.*]]
// LLVM: [[SCOPE]]:
// LLVM:   store ptr @.str, ptr %[[REF_TMP]]
// LLVM:   call void @_ZNSt6vectorIPKcE9push_backEOS1_(ptr %[[RESULT]], ptr %[[REF_TMP]])
// LLVM:   br label %[[DONE:.*]]
// LLVM: [[DONE]]:
// LLVM:   store i8 1, ptr %[[NRVO_FLAG]]
// LLVM:   %[[NRVO_FLAG_VAL:.*]] = load i8, ptr %[[NRVO_FLAG]]
// LLVM:   %[[NRVO_FLAG_BOOL:.*]] = trunc i8 %[[NRVO_FLAG_VAL]] to i1
// LLVM:   %[[NOT_NRVO:.*]] = xor i1 %[[NRVO_FLAG_BOOL]], true
// LLVM:   br i1 %[[NOT_NRVO]], label %[[NRVO_UNUSED:.*]], label %[[END:.*]]
// LLVM: [[NRVO_UNUSED]]:
// LLVM:   call void @_ZNSt6vectorIPKcED1Ev(ptr %[[RESULT]])
// LLVM:   br label %[[END]]
// LLVM: [[END]]:
// LLVM:   %[[RETVAL:.*]] = load %[[VEC]], ptr %[[RESULT]]
// LLVM:   ret %[[VEC]] %[[RETVAL]]

// OGCG: define {{.*}} void @_Z9test_nrvov(ptr {{.*}} sret(%[[VEC:.*]]) {{.*}} %[[RESULT:.*]])
// OGCG:   %[[RESULT_ADDR:.*]] = alloca ptr
// OGCG:   %[[NRVO_FLAG:.*]] = alloca i1
// OGCG:   %[[REF_TMP:.*]] = alloca ptr
// OGCG:   store ptr %[[RESULT]], ptr %[[RESULT_ADDR]]
// OGCG:   store i1 false, ptr %[[NRVO_FLAG]]
// OGCG:   call void @_ZNSt6vectorIPKcEC1Ev(ptr {{.*}} %[[RESULT]])
// OGCG:   store ptr @.str, ptr %[[REF_TMP]]
// OGCG:   call void @_ZNSt6vectorIPKcE9push_backEOS1_(ptr {{.*}} %[[RESULT]], ptr {{.*}} %[[REF_TMP]])
// OGCG:   store i1 true, ptr %[[NRVO_FLAG]]
// OGCG:   %[[NRVO_FLAG_VAL:.*]] = load i1, ptr %[[NRVO_FLAG]]
// OGCG:   br i1 %[[NRVO_FLAG_VAL]], label %[[END:.*]], label %[[NRVO_UNUSED:.*]]
// OGCG: [[NRVO_UNUSED]]:
// OGCG:   call void @_ZNSt6vectorIPKcED1Ev(ptr {{.*}} %[[RESULT]])
// OGCG:   br label %[[END]]
// OGCG: [[END]]
// OGCG:   ret void
