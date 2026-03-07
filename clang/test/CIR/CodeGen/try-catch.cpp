// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

void empty_try_block_with_catch_all() {
  try {} catch (...) {}
}

// CIR: cir.func{{.*}} @_Z30empty_try_block_with_catch_allv()
// CIR:   cir.return

// LLVM: define{{.*}} void @_Z30empty_try_block_with_catch_allv()
// LLVM:  ret void

// OGCG: define{{.*}} void @_Z30empty_try_block_with_catch_allv()
// OGCG:   ret void

void empty_try_block_with_catch_with_int_exception() {
  try {} catch (int e) {}
}

// CIR: cir.func{{.*}} @_Z45empty_try_block_with_catch_with_int_exceptionv()
// CIR:   cir.return

// LLVM: define{{.*}} void @_Z45empty_try_block_with_catch_with_int_exceptionv()
// LLVM:  ret void

// OGCG: define{{.*}} void @_Z45empty_try_block_with_catch_with_int_exceptionv()
// OGCG:   ret void

void try_catch_with_empty_catch_all() {
  int a = 1;
  try {
    return;
    ++a;
  } catch (...) {
  }
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CIR: %[[CONST_1:.*]] = cir.const #cir.int<1> : !s32i
// CIR: cir.store{{.*}} %[[CONST_1]], %[[A_ADDR]] : !s32i, !cir.ptr<!s32i
// CIR: cir.scope {
// CIR:   cir.try {
// CIR:     cir.return
// CIR:   ^bb1:  // no predecessors
// CIR:     %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:     %[[RESULT:.*]] = cir.unary(inc, %[[TMP_A]]) nsw : !s32i, !s32i
// CIR:     cir.store{{.*}} %[[RESULT]], %[[A_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:     cir.yield
// CIR:   }
// CIR: }

// LLVM:   %[[A_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM:   store i32 1, ptr %[[A_ADDR]], align 4
// LLVM:   br label %[[BB_2:.*]]
// LLVM: [[BB_2]]:
// LLVM:   br label %[[BB_3:.*]]
// LLVM: [[BB_3]]:
// LLVM:   ret void
// LLVM: [[BB_4:.*]]:
// LLVM:   %[[TMP_A:.*]] = load i32, ptr %[[A_ADDR]], align 4
// LLVM:   %[[RESULT:.*]] = add nsw i32 %[[TMP_A]], 1
// LLVM:   store i32 %[[RESULT]], ptr %[[A_ADDR]], align 4
// LLVM:   br label %[[BB_7:.*]]
// LLVM: [[BB_7]]:
// LLVM:   br label %[[BB_8:.*]]
// LLVM: [[BB_8]]:
// LLVM:   ret void

// OGCG: %[[A_ADDR:.*]] = alloca i32, align 4
// OGCG: store i32 1, ptr %[[A_ADDR]], align 4
// OGCG: ret void

void try_catch_with_empty_catch_all_2() {
  int a = 1;
  try {
    ++a;
    return;
  } catch (...) {
  }
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init]
// CIR: %[[CONST_1:.*]] = cir.const #cir.int<1> : !s32i
// CIR: cir.store{{.*}} %[[CONST_1]], %[[A_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR: cir.scope {
// CIR:   cir.try {
// CIR:     %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:     %[[RESULT:.*]] = cir.unary(inc, %[[TMP_A]]) nsw : !s32i, !s32i
// CIR:     cir.store{{.*}} %[[RESULT]], %[[A_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:     cir.return
// CIR:   }
// CIR: }

// LLVM:   %[[A_ADDR]] = alloca i32, i64 1, align 4
// LLVM:   store i32 1, ptr %[[A_ADDR]], align 4
// LLVM:   br label %[[BB_2:.*]]
// LLVM: [[BB_2]]:
// LLVM:   br label %[[BB_3:.*]]
// LLVM: [[BB_3]]:
// LLVM:   %[[TMP_A:.*]] = load i32, ptr %[[A_ADDR]], align 4
// LLVM:   %[[RESULT:.*]] = add nsw i32 %[[TMP_A:.*]], 1
// LLVM:   store i32 %[[RESULT]], ptr %[[A_ADDR]], align 4
// LLVM:   ret void
// LLVM: [[BB_6:.*]]:
// LLVM:   br label %[[BB_7:.*]]
// LLVM: [[BB_7]]:
// LLVM:   ret void

// OGCG: %[[A_ADDR:.*]] = alloca i32, align 4
// OGCG: store i32 1, ptr %[[A_ADDR]], align 4
// OGCG: %[[TMP_A:.*]] = load i32, ptr %[[A_ADDR]], align 4
// OGCG: %[[RESULT:.*]] = add nsw i32 %[[TMP_A]], 1
// OGCG: store i32 %[[RESULT]], ptr %[[A_ADDR]], align 4
// OGCG: ret void

void try_catch_with_alloca() {
  try {
    int a;
    int b;
    int c = a + b;
  } catch (...) {
  }
}

// CIR: cir.scope {
// CIR:   %[[A_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a"]
// CIR:   %[[B_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["b"]
// CIR:   %[[C_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["c", init]
// CIR:   cir.try {
// CIR:     %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:     %[[TMP_B:.*]] = cir.load{{.*}} %[[B_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:     %[[RESULT:.*]] = cir.add nsw %[[TMP_A]], %[[TMP_B]] : !s32i
// CIR:     cir.store{{.*}} %[[RESULT]], %[[C_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:     cir.yield
// CIR:   }
// CIR: }

// LLVM:  %[[A_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM:  %[[B_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM:  %[[C_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM:  br label %[[LABEL_1:.*]]
// LLVM: [[LABEL_1]]:
// LLVM:  br label %[[LABEL_2:.*]]
// LLVM: [[LABEL_2]]:
// LLVM:  %[[TMP_A:.*]] = load i32, ptr %[[A_ADDR]], align 4
// LLVM:  %[[TMP_B:.*]] = load i32, ptr %[[B_ADDR]], align 4
// LLVM:  %[[RESULT:.*]] = add nsw i32 %[[TMP_A]], %[[TMP_B]]
// LLVM:  store i32 %[[RESULT]], ptr %[[C_ADDR]], align 4
// LLVM:  br label %[[LABEL_3:.*]]
// LLVM: [[LABEL_3]]:
// LLVM:  br label %[[LABEL_4:.*]]
// LLVM: [[LABEL_4]]:
// LLVM:  ret void

// OGCG: %[[A_ADDR:.*]] = alloca i32, align 4
// OGCG: %[[B_ADDR:.*]] = alloca i32, align 4
// OGCG: %[[C_ADDR:.*]] = alloca i32, align 4
// OGCG: %[[TMP_A:.*]] = load i32, ptr %[[A_ADDR]], align 4
// OGCG: %[[TMP_B:.*]] = load i32, ptr %[[B_ADDR]], align 4
// OGCG: %[[RESULT:.*]] = add nsw i32 %[[TMP_A]], %[[TMP_B]]
// OGCG: store i32 %[[RESULT]], ptr %[[C_ADDR]], align 4

void function_with_noexcept() noexcept;

void calling_noexcept_function_inside_try_block() {
  try {
    function_with_noexcept();
  } catch (...) {
  }
}

// CIR: cir.scope {
// CIR:   cir.try {
// CIR:     cir.call @_Z22function_with_noexceptv() nothrow : () -> ()
// CIR:     cir.yield
// CIR:   }
// CIR: }

// LLVM:   br label %[[LABEL_1:.*]]
// LLVM: [[LABEL_1]]:
// LLVM:   br label %[[LABEL_2:.*]]
// LLVM: [[LABEL_2]]:
// LLVM:   call void @_Z22function_with_noexceptv()
// LLVM:   br label %[[LABEL_3:.*]]
// LLVM: [[LABEL_3]]:
// LLVM:   br label %[[LABEL_4:.*]]
// LLVM: [[LABEL_4]]:
// LLVM:   ret void

// OGCG: call void @_Z22function_with_noexceptv()
// OGCG: ret void

int division();

void call_function_inside_try_catch_all() {
  try {
    division();
  } catch (...) {
  }
}

// CIR: cir.func {{.*}} @_Z34call_function_inside_try_catch_allv() personality(@__gxx_personality_v0)
// CIR: cir.scope {
// CIR:   cir.try {
// CIR:       %[[CALL:.*]] = cir.call @_Z8divisionv()
// CIR:       cir.yield
// CIR:   } catch all (%[[EH_TOKEN:.*]]: !cir.eh_token{{.*}}) {
// CIR:       %[[CATCH_TOKEN:.*]], %[[EXN_PTR:.*]] = cir.begin_catch %[[EH_TOKEN]] : !cir.eh_token -> (!cir.catch_token, !cir.ptr<!void>)
// CIR:       cir.cleanup.scope {
// CIR:         cir.yield
// CIR:       } cleanup {{.*}} {
// CIR:         cir.end_catch %[[CATCH_TOKEN]] : !cir.catch_token
// CIR:         cir.yield
// CIR:       }
// CIR:       cir.yield
// CIR:   }
// CIR: }

// LLVM: define {{.*}} void @_Z34call_function_inside_try_catch_allv() {{.*}} personality ptr @__gxx_personality_v0
// LLVM:   br label %[[TRY_SCOPE:.*]]
// LLVM: [[TRY_SCOPE]]:
// LLVM:   br label %[[TRY_BEGIN:.*]]
// LLVM: [[TRY_BEGIN]]:
// LLVM:   %[[CALL:.*]] = invoke i32 @_Z8divisionv()
// LLVM:           to label %[[INVOKE_CONT:.*]] unwind label %[[LANDING_PAD:.*]]
// LLVM: [[INVOKE_CONT]]:
// LLVM:   br label %[[TRY_CONT:.*]]
// LLVM: [[LANDING_PAD]]:
// LLVM:   %[[LP:.*]] = landingpad { ptr, i32 }
// LLVM:           catch ptr null
// LLVM:   %[[EXN_OBJ:.*]] = extractvalue { ptr, i32 } %[[LP]], 0
// LLVM:   %[[EH_SELECTOR_VAL:.*]] = extractvalue { ptr, i32 } %[[LP]], 1
// LLVM:   br label %[[CATCH:.*]]
// LLVM: [[CATCH]]:
// LLVM:   %[[EXN_OBJ_PHI:.*]] = phi ptr [ %[[EXN_OBJ:.*]], %[[LANDING_PAD:.*]] ]
// LLVM:   %[[EH_SELECTOR_PHI:.*]] = phi i32 [ %[[EH_SELECTOR_VAL:.*]], %[[LANDING_PAD:.*]] ]
// LLVM:   br label %[[BEGIN_CATCH:.*]]
// LLVM: [[BEGIN_CATCH]]:
// LLVM:   %[[EXN_OBJ_PHI1:.*]] = phi ptr [ %[[EXN_OBJ:.*]], %[[LANDING_PAD:.*]] ]
// LLVM:   %[[EH_SELECTOR_PHI1:.*]] = phi i32 [ %[[EH_SELECTOR_VAL:.*]], %[[LANDING_PAD:.*]] ]
// LLVM:   %[[TOKEN:.*]] = call ptr @__cxa_begin_catch(ptr %[[EXN_OBJ_PHI1]])
// LLVM:   br label %[[CATCH_BODY:.*]]
// LLVM: [[CATCH_BODY]]:
// LLVM:   br label %[[END_CATCH:.*]]
// LLVM: [[END_CATCH]]:
// LLVM:   call void @__cxa_end_catch()
// LLVM:   br label %[[END_DISPATCH:.*]]
// LLVM: [[END_DISPATCH]]:
// LLVM:   br label %[[END_TRY:.*]]
// LLVM: [[END_TRY]]:
// LLVM:   br label %[[TRY_CONT:.*]]
// LLVM: [[TRY_CONT]]:
// LLVM:   br label %[[DONE:.*]]
// LLVM: [[DONE]]:
// LLVM:   ret void

// OGCG: define {{.*}} void @_Z34call_function_inside_try_catch_allv() {{.*}} personality ptr @__gxx_personality_v0
// OGCG:   %[[EXN_OBJ_ADDR:.*]] = alloca ptr, align 8
// OGCG:   %[[EH_SELECTOR_ADDR:.*]] = alloca i32, align 4
// OGCG:   %[[CALL:.*]] = invoke noundef i32 @_Z8divisionv()
// OGCG:           to label %[[INVOKE_CONT:.*]] unwind label %[[LANDING_PAD:.*]]
// OGCG: [[INVOKE_CONT]]:
// OGCG:   br label %[[TRY_CONT:.*]]
// OGCG: [[LANDING_PAD]]:
// OGCG:   %[[LP:.*]] = landingpad { ptr, i32 }
// OGCG:           catch ptr null
// OGCG:   %[[EXN_OBJ:.*]] = extractvalue { ptr, i32 } %[[LP]], 0
// OGCG:   store ptr %[[EXN_OBJ]], ptr %[[EXN_OBJ_ADDR]], align 8
// OGCG:   %[[EH_SELECTOR_VAL:.*]] = extractvalue { ptr, i32 } %[[LP]], 1
// OGCG:   store i32 %[[EH_SELECTOR_VAL]], ptr %[[EH_SELECTOR_ADDR]], align 4
// OGCG:   br label %[[CATCH:.*]]
// OGCG: [[CATCH]]:
// OGCG:   %[[EXN_OBJ:.*]] = load ptr, ptr %[[EXN_OBJ_ADDR]], align 8
// OGCG:   %[[CATCH_BEGIN:.*]] = call ptr @__cxa_begin_catch(ptr %[[EXN_OBJ]])
// OGCG:   call void @__cxa_end_catch()
// OGCG:   br label %[[TRY_CONT]]
// OGCG: [[TRY_CONT]]:
// OGCG:   ret void

void call_function_inside_try_catch_with_exception_type() {
  try {
    division();
  } catch (int e) {
  }
}

// CIR: cir.func {{.*}} @_Z50call_function_inside_try_catch_with_exception_typev() personality(@__gxx_personality_v0)
// CIR: cir.scope {
// CIR:   %[[EXCEPTION_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["e"]
// CIR:   cir.try {
// CIR:     %[[CALL:.*]] = cir.call @_Z8divisionv()
// CIR:     cir.yield
// CIR:   } catch [type #cir.global_view<@_ZTIi> : !cir.ptr<!u8i>] (%[[EH_TOKEN:.*]]: !cir.eh_token{{.*}}) {
// CIR:     %[[CATCH_TOKEN:.*]], %[[EXN_PTR:.*]] = cir.begin_catch %[[EH_TOKEN]] : !cir.eh_token -> (!cir.catch_token, !cir.ptr<!s32i>)
// CIR:     cir.cleanup.scope {
// CIR:       %[[TMP:.*]] = cir.load {{.*}} %[[EXN_PTR]] : !cir.ptr<!s32i>, !s32i
// CIR:       cir.store {{.*}} %[[TMP]], %[[EXCEPTION_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:       cir.yield
// CIR:     } cleanup {{.*}} {
// CIR:       cir.end_catch %[[CATCH_TOKEN]] : !cir.catch_token
// CIR:       cir.yield
// CIR:     }
// CIR:     cir.yield
// CIR:   } unwind (%{{.*}}: !cir.eh_token{{.*}}) {
// CIR:     cir.resume %{{.*}} : !cir.eh_token
// CIR:   }
// CIR: }

// LLVM: define {{.*}} void @_Z50call_function_inside_try_catch_with_exception_typev() {{.*}} personality ptr @__gxx_personality_v0
// LLVM:   br label %[[TRY_SCOPE:.*]]
// LLVM: [[TRY_SCOPE]]:
// LLVM:   br label %[[TRY_BEGIN:.*]]
// LLVM: [[TRY_BEGIN]]:
// LLVM:   %[[CALL:.*]] = invoke i32 @_Z8divisionv()
// LLVM:           to label %[[INVOKE_CONT:.*]] unwind label %[[LANDING_PAD:.*]]
// LLVM: [[INVOKE_CONT]]:
// LLVM:   br label %[[TRY_CONT:.*]]
// LLVM: [[LANDING_PAD]]:
// LLVM:   %[[LP:.*]] = landingpad { ptr, i32 }
// LLVM:           catch ptr @_ZTIi
// LLVM:   %[[EXN_OBJ:.*]] = extractvalue { ptr, i32 } %[[LP]], 0
// LLVM:   %[[EH_SELECTOR_VAL:.*]] = extractvalue { ptr, i32 } %[[LP]], 1
// LLVM:   br label %[[CATCH:.*]]
// LLVM: [[CATCH]]:
// LLVM:   %[[EXN_OBJ_PHI:.*]] = phi ptr [ %[[EXN_OBJ:.*]], %[[LANDING_PAD:.*]] ]
// LLVM:   %[[EH_SELECTOR_PHI:.*]] = phi i32 [ %[[EH_SELECTOR_VAL:.*]], %[[LANDING_PAD:.*]] ]
// LLVM:   br label %[[DISPATCH:.*]]
// LLVM: [[DISPATCH]]:
// LLVM:   %[[EXN_OBJ_PHI1:.*]] = phi ptr [ %[[EXN_OBJ_PHI:.*]], %[[CATCH:.*]] ]
// LLVM:   %[[EH_SELECTOR_PHI1:.*]] = phi i32 [ %[[EH_SELECTOR_PHI:.*]], %[[CATCH:.*]] ]
// LLVM:   %[[EH_TYPE_ID:.*]] = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
// LLVM:   %[[TYPE_ID_EQ:.*]] = icmp eq i32 %[[EH_SELECTOR_PHI1]], %[[EH_TYPE_ID]]
// LLVM:   br i1 %[[TYPE_ID_EQ]], label %[[BEGIN_CATCH:.*]], label %[[RESUME:.*]]
// LLVM: [[BEGIN_CATCH]]:
// LLVM:   %[[EXN_OBJ_PHI2:.*]] = phi ptr [ %[[EXN_OBJ_PHI1:.*]], %[[DISPATCH:.*]] ]
// LLVM:   %[[EH_SELECTOR_PHI2:.*]] = phi i32 [ %[[EH_SELECTOR_PHI1:.*]], %[[DISPATCH:.*]] ]
// LLVM:   %[[TOKEN:.*]] = call ptr @__cxa_begin_catch(ptr %[[EXN_OBJ_PHI2]])
// LLVM:   br label %[[CATCH_BODY:.*]]
// LLVM: [[CATCH_BODY]]:
// LLVM:   %[[LOAD:.*]] = load i32, ptr %[[TOKEN]], align 4
// LLVM:   store i32 %[[LOAD]], ptr {{.*}}, align 4
// LLVM:   br label %[[END_CATCH:.*]]
// LLVM: [[END_CATCH]]:
// LLVM:   call void @__cxa_end_catch()
// LLVM:   br label %[[END_DISPATCH:.*]]
// LLVM: [[END_DISPATCH]]:
// LLVM:   br label %[[END_TRY:.*]]
// LLVM: [[END_TRY]]:
// LLVM:   br label %[[TRY_CONT:.*]]
// LLVM: [[RESUME]]:
// LLVM:   %[[EXN_OBJ_PHI3:.*]] = phi ptr [ %[[EXN_OBJ_PHI1:.*]], %[[DISPATCH:.*]] ]
// LLVM:   %[[EH_SELECTOR_PHI3:.*]] = phi i32 [ %[[EH_SELECTOR_PHI1:.*]], %[[DISPATCH:.*]] ]
// LLVM:   %[[TMP_EXCEPTION_INFO:.*]] = insertvalue { ptr, i32 } poison, ptr %[[EXN_OBJ_PHI3]], 0
// LLVM:   %[[EXCEPTION_INFO:.*]] = insertvalue { ptr, i32 } %[[TMP_EXCEPTION_INFO]], i32 %[[EH_SELECTOR_PHI3]], 1
// LLVM:   resume { ptr, i32 } %[[EXCEPTION_INFO]]
// LLVM: [[TRY_CONT]]:
// LLVM:   br label %[[DONE:.*]]
// LLVM: [[DONE]]:
// LLVM:   ret void

// OGCG: define {{.*}} void @_Z50call_function_inside_try_catch_with_exception_typev() {{.*}} personality ptr @__gxx_personality_v0
// OGCG:   %[[EXCEPTION_ADDR:.*]] = alloca ptr, align 8
// OGCG:   %[[EH_TYPE_ID_ADDR:.*]] = alloca i32, align 4
// OGCG:   %[[E_ADDR:.*]] = alloca i32, align 4
// OGCG:   %[[CALL:.*]] = invoke noundef i32 @_Z8divisionv()
// OGCG:           to label %[[INVOKE_NORMAL:.*]] unwind label %[[INVOKE_UNWIND:.*]]
// OGCG: [[INVOKE_NORMAL]]:
// OGCG:   br label %[[TRY_CONT:.*]]
// OGCG: [[INVOKE_UNWIND]]:
// OGCG:   %[[LANDING_PAD:.*]] = landingpad { ptr, i32 }
// OGCG:           catch ptr @_ZTIi
// OGCG:   %[[EXCEPTION:.*]] = extractvalue { ptr, i32 } %[[LANDING_PAD]], 0
// OGCG:   store ptr %[[EXCEPTION]], ptr %[[EXCEPTION_ADDR]], align 8
// OGCG:   %[[EH_TYPE_ID:.*]] = extractvalue { ptr, i32 } %[[LANDING_PAD]], 1
// OGCG:   store i32 %[[EH_TYPE_ID]], ptr %[[EH_TYPE_ID_ADDR]], align 4
// OGCG:   br label %[[CATCH_DISPATCH:.*]]
// OGCG: [[CATCH_DISPATCH]]:
// OGCG:   %[[TMP_EH_TYPE_ID:.*]] = load i32, ptr %[[EH_TYPE_ID_ADDR]], align 4
// OGCG:   %[[EH_TYPE_ID:.*]] = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
// OGCG:   %[[TYPE_ID_EQ:.*]] = icmp eq i32 %[[TMP_EH_TYPE_ID]], %[[EH_TYPE_ID]]
// OGCG:   br i1 %[[TYPE_ID_EQ]], label %[[CATCH_EXCEPTION:.*]], label %[[EH_RESUME:.*]]
// OGCG: [[CATCH_EXCEPTION]]:
// OGCG:   %[[TMP_EXCEPTION:.*]] = load ptr, ptr %[[EXCEPTION_ADDR]], align 8
// OGCG:   %[[BEGIN_CATCH:.*]] = call ptr @__cxa_begin_catch(ptr %[[TMP_EXCEPTION]])
// OGCG:   %[[TMP_BEGIN_CATCH:.*]] = load i32, ptr %[[BEGIN_CATCH]], align 4
// OGCG:   store i32 %[[TMP_BEGIN_CATCH]], ptr %[[E_ADDR]], align 4
// OGCG:   call void @__cxa_end_catch()
// OGCG:   br label %[[TRY_CONT]]
// OGCG: [[TRY_CONT]]:
// OGCG:   ret void
// OGCG: [[EH_RESUME]]:
// OGCG:   %[[TMP_EXCEPTION:.*]] = load ptr, ptr %[[EXCEPTION_ADDR]], align 8
// OGCG:   %[[TMP_EH_TYPE_ID:.*]] = load i32, ptr %[[EH_TYPE_ID_ADDR]], align 4
// OGCG:   %[[TMP_EXCEPTION_INFO:.*]] = insertvalue { ptr, i32 } poison, ptr %[[TMP_EXCEPTION]], 0
// OGCG:   %[[EXCEPTION_INFO:.*]] = insertvalue { ptr, i32 } %[[TMP_EXCEPTION_INFO]], i32 %[[TMP_EH_TYPE_ID]], 1
// OGCG:   resume { ptr, i32 } %[[EXCEPTION_INFO]]

void call_function_inside_try_catch_with_ref_exception_type() {
  try {
    division();
  } catch (int &ref) {
  }
}

// CIR: cir.func {{.*}} @_Z54call_function_inside_try_catch_with_ref_exception_typev() personality(@__gxx_personality_v0)
// CIR: cir.scope {
// CIR:   %[[EXCEPTION_ADDR:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["ref", const]
// CIR:   cir.try {
// CIR:     %[[CALL:.*]] = cir.call @_Z8divisionv()
// CIR:     cir.yield
// CIR:   } catch [type #cir.global_view<@_ZTIi> : !cir.ptr<!u8i>] (%{{.*}}: !cir.eh_token {{.*}}) {
// CIR:     %[[CATCH_TOKEN:.*]], %[[EXN_PTR:.*]] = cir.begin_catch %{{.*}} : !cir.eh_token -> (!cir.catch_token, !cir.ptr<!s32i>)
// CIR:     cir.cleanup.scope {
// CIR:       cir.store {{.*}} %[[EXN_PTR]], %[[EXCEPTION_ADDR]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR:       cir.yield
// CIR:     } cleanup all {
// CIR:       cir.end_catch %[[CATCH_TOKEN]] : !cir.catch_token
// CIR:       cir.yield
// CIR:     }
// CIR:     cir.yield
// CIR:   } unwind (%{{.*}}: !cir.eh_token {{.*}}) {
// CIR:     cir.resume %{{.*}} : !cir.eh_token
// CIR:   }
// CIR: }

// LLVM: define {{.*}} void @_Z54call_function_inside_try_catch_with_ref_exception_typev() {{.*}} personality ptr @__gxx_personality_v0
// LLVM:   br label %[[TRY_SCOPE:.*]]
// LLVM: [[TRY_SCOPE]]:
// LLVM:   br label %[[TRY_BEGIN:.*]]
// LLVM: [[TRY_BEGIN]]:
// LLVM:   invoke i32 @_Z8divisionv()
// LLVM:           to label %[[INVOKE_CONT:.*]] unwind label %[[LANDING_PAD:.*]]
// LLVM: [[INVOKE_CONT:.*]]:
// LLVM:   br label %[[TRY_CONT:.*]]
// LLVM: [[LANDING_PAD]]:
// LLVM:   %[[LP:.*]] = landingpad { ptr, i32 }
// LLVM:           catch ptr @_ZTIi
// LLVM:   %[[EXN_OBJ:.*]] = extractvalue { ptr, i32 } %[[LP]], 0
// LLVM:   %[[EH_SELECTOR_VAL:.*]] = extractvalue { ptr, i32 } %[[LP]], 1
// LLVM:   br label %[[CATCH:.*]]
// LLVM: [[CATCH]]:
// LLVM:   %[[EXN_OBJ_PHI:.*]] = phi ptr [ %[[EXN_OBJ:.*]], %[[LANDING_PAD:.*]] ]
// LLVM:   %[[EH_SELECTOR_PHI:.*]] = phi i32 [ %[[EH_SELECTOR_VAL:.*]], %[[LANDING_PAD:.*]] ]
// LLVM:   br label %[[DISPATCH:.*]]
// LLVM: [[DISPATCH]]:
// LLVM:   %[[EXN_OBJ_PHI1:.*]] = phi ptr [ %[[EXN_OBJ_PHI:.*]], %[[CATCH:.*]] ]
// LLVM:   %[[EH_SELECTOR_PHI1:.*]] = phi i32 [ %[[EH_SELECTOR_PHI:.*]], %[[CATCH:.*]] ]
// LLVM:   %[[EH_TYPE_ID:.*]] = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
// LLVM:   %[[TYPE_ID_EQ:.*]] = icmp eq i32 %[[EH_SELECTOR_PHI1]], %[[EH_TYPE_ID]]
// LLVM:   br i1 %[[TYPE_ID_EQ]], label %[[BEGIN_CATCH:.*]], label %[[RESUME:.*]]
// LLVM: [[BEGIN_CATCH]]:
// LLVM:   %[[EXN_OBJ_PHI2:.*]] = phi ptr [ %[[EXN_OBJ_PHI1:.*]], %[[DISPATCH:.*]] ]
// LLVM:   %[[EH_SELECTOR_PHI2:.*]] = phi i32 [ %[[EH_SELECTOR_PHI1:.*]], %[[DISPATCH:.*]] ]
// LLVM:   %[[TOKEN:.*]] = call ptr @__cxa_begin_catch(ptr %[[EXN_OBJ_PHI2]])
// LLVM:   br label %[[CATCH_BODY:.*]]
// LLVM: [[CATCH_BODY]]:
// LLVM:   store ptr %[[TOKEN]], ptr %{{.*}}, align 8
// LLVM:   br label %[[END_CATCH:.*]]
// LLVM: [[END_CATCH]]:
// LLVM:   call void @__cxa_end_catch()
// LLVM:   br label %[[END_DISPATCH:.*]]
// LLVM: [[END_DISPATCH]]:
// LLVM:   br label %[[END_TRY:.*]]
// LLVM: [[END_TRY]]:
// LLVM:   br label %[[TRY_CONT:.*]]
// LLVM: [[RESUME]]:
// LLVM:   %[[EXN_OBJ_PHI3:.*]] = phi ptr [ %[[EXN_OBJ_PHI1:.*]], %[[DISPATCH:.*]] ]
// LLVM:   %[[EH_SELECTOR_PHI3:.*]] = phi i32 [ %[[EH_SELECTOR_PHI1:.*]], %[[DISPATCH:.*]] ]
// LLVM:   %[[TMP_EXCEPTION_INFO:.*]] = insertvalue { ptr, i32 } poison, ptr %[[EXN_OBJ_PHI3]], 0
// LLVM:   %[[EXCEPTION_INFO:.*]] = insertvalue { ptr, i32 } %[[TMP_EXCEPTION_INFO]], i32 %[[EH_SELECTOR_PHI3]], 1
// LLVM:   resume { ptr, i32 } %[[EXCEPTION_INFO]]
// LLVM: [[TRY_CONT]]:
// LLVM:   br label %[[DONE:.*]]
// LLVM: [[DONE]]:
// LLVM:   ret void

// OGCG: define {{.*}} void @_Z54call_function_inside_try_catch_with_ref_exception_typev() {{.*}} personality ptr @__gxx_personality_v0
// OGCG:   %[[EXCEPTION_ADDR:.*]] = alloca ptr, align 8
// OGCG:   %[[EH_TYPE_ID_ADDR:.*]] = alloca i32, align 4
// OGCG:   %[[E_ADDR:.*]] = alloca ptr, align 8
// OGCG:   %[[CALL:.*]] = invoke noundef i32 @_Z8divisionv()
// OGCG:           to label %[[INVOKE_NORMAL:.*]] unwind label %[[INVOKE_UNWIND:.*]]
// OGCG: [[INVOKE_NORMAL]]:
// OGCG:   br label %[[TRY_CONT:.*]]
// OGCG: [[INVOKE_UNWIND]]:
// OGCG:   %[[LANDING_PAD:.*]] = landingpad { ptr, i32 }
// OGCG:           catch ptr @_ZTIi
// OGCG:   %[[EXCEPTION:.*]] = extractvalue { ptr, i32 } %[[LANDING_PAD]], 0
// OGCG:   store ptr %1, ptr %[[EXCEPTION_ADDR]], align 8
// OGCG:   %[[EH_TYPE_ID:.*]] = extractvalue { ptr, i32 } %[[LANDING_PAD]], 1
// OGCG:   store i32 %[[EH_TYPE_ID]], ptr %[[EH_TYPE_ID_ADDR]], align 4
// OGCG:   br label %[[CATCH_DISPATCH:.*]]
// OGCG: [[CATCH_DISPATCH]]:
// OGCG:   %[[TMP_EH_TYPE_ID:.*]] = load i32, ptr %[[EH_TYPE_ID_ADDR]], align 4
// OGCG:   %[[EH_TYPE_ID:.*]] = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
// OGCG:   %[[TYPE_ID_EQ:.*]] = icmp eq i32 %[[TMP_EH_TYPE_ID]], %[[EH_TYPE_ID]]
// OGCG:   br i1 %[[TYPE_ID_EQ]], label %[[CATCH_EXCEPTION:.*]], label %[[EH_RESUME:.*]]
// OGCG: [[CATCH_EXCEPTION]]:
// OGCG:   %[[TMP_EXCEPTION:.*]] = load ptr, ptr %[[EXCEPTION_ADDR]], align 8
// OGCG:   %[[BEGIN_CATCH:.*]] = call ptr @__cxa_begin_catch(ptr %[[TMP_EXCEPTION]])
// OGCG:   store ptr %[[BEGIN_CATCH]], ptr %[[E_ADDR]], align 8
// OGCG:   call void @__cxa_end_catch()
// OGCG:   br label %[[TRY_CONT]]
// OGCG: [[TRY_CONT]]:
// OGCG:   ret void
// OGCG: [[EH_RESUME]]:
// OGCG:   %[[TMP_EXCEPTION:.*]] = load ptr, ptr %[[EXCEPTION_ADDR]], align 8
// OGCG:   %[[TMP_EH_TYPE_ID:.*]] = load i32, ptr %[[EH_TYPE_ID_ADDR]], align 4
// OGCG:   %[[TMP_EXCEPTION_INFO:.*]] = insertvalue { ptr, i32 } poison, ptr %[[TMP_EXCEPTION]], 0
// OGCG:   %[[EXCEPTION_INFO:.*]] = insertvalue { ptr, i32 } %[[TMP_EXCEPTION_INFO]], i32 %[[TMP_EH_TYPE_ID]], 1
// OGCG:   resume { ptr, i32 } %[[EXCEPTION_INFO]]

void call_function_inside_try_catch_with_complex_exception_type() {
  try {
    division();
  } catch (int _Complex e) {
  }
}

// CIR: cir.func {{.*}} @_Z58call_function_inside_try_catch_with_complex_exception_typev() personality(@__gxx_personality_v0)
// CIR: cir.scope {
// CIR:   %[[EXCEPTION_ADDR:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["e"]
// CIR:   cir.try {
// CIR:     %[[CALL:.*]] = cir.call @_Z8divisionv()
// CIR:     cir.yield
// CIR:   } catch [type #cir.global_view<@_ZTICi> : !cir.ptr<!u8i>] (%[[EH_TOKEN:.*]]: !cir.eh_token{{.*}}) {
// CIR:     %[[CATCH_TOKEN:.*]], %[[EXN_PTR:.*]] = cir.begin_catch %[[EH_TOKEN]] : !cir.eh_token -> (!cir.catch_token, !cir.ptr<!cir.complex<!s32i>>)
// CIR:     cir.cleanup.scope {
// CIR:       %[[TMP:.*]] = cir.load {{.*}} %[[EXN_PTR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR:       cir.store {{.*}} %[[TMP]], %[[EXCEPTION_ADDR]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>
// CIR:       cir.yield
// CIR:     } cleanup {{.*}} {
// CIR:       cir.end_catch %[[CATCH_TOKEN]] : !cir.catch_token
// CIR:       cir.yield
// CIR:     }
// CIR:     cir.yield
// CIR:   } unwind (%{{.*}}: !cir.eh_token{{.*}}) {
// CIR:     cir.resume %{{.*}} : !cir.eh_token
// CIR:   }
// CIR: }

// LLVM: define {{.*}} void @_Z58call_function_inside_try_catch_with_complex_exception_typev() {{.*}} personality ptr @__gxx_personality_v0
// LLVM:   br label %[[TRY_SCOPE:.*]]
// LLVM: [[TRY_SCOPE]]:
// LLVM:   br label %[[TRY_BEGIN:.*]]
// LLVM: [[TRY_BEGIN]]:
// LLVM:   %[[CALL:.*]] = invoke i32 @_Z8divisionv()
// LLVM:           to label %[[INVOKE_CONT:.*]] unwind label %[[LANDING_PAD:.*]]
// LLVM: [[INVOKE_CONT]]:
// LLVM:   br label %[[TRY_CONT:.*]]
// LLVM: [[LANDING_PAD]]:
// LLVM:   %[[LP:.*]] = landingpad { ptr, i32 }
// LLVM:           catch ptr @_ZTICi
// LLVM:   %[[EXN_OBJ:.*]] = extractvalue { ptr, i32 } %[[LP]], 0
// LLVM:   %[[EH_SELECTOR_VAL:.*]] = extractvalue { ptr, i32 } %[[LP]], 1
// LLVM:   br label %[[CATCH:.*]]
// LLVM: [[CATCH]]:
// LLVM:   %[[EXN_OBJ_PHI:.*]] = phi ptr [ %[[EXN_OBJ:.*]], %[[LANDING_PAD:.*]] ]
// LLVM:   %[[EH_SELECTOR_PHI:.*]] = phi i32 [ %[[EH_SELECTOR_VAL:.*]], %[[LANDING_PAD:.*]] ]
// LLVM:   br label %[[DISPATCH:.*]]
// LLVM: [[DISPATCH]]:
// LLVM:   %[[EXN_OBJ_PHI1:.*]] = phi ptr [ %[[EXN_OBJ_PHI:.*]], %[[CATCH:.*]] ]
// LLVM:   %[[EH_SELECTOR_PHI1:.*]] = phi i32 [ %[[EH_SELECTOR_PHI:.*]], %[[CATCH:.*]] ]
// LLVM:   %[[EH_TYPE_ID:.*]] = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTICi)
// LLVM:   %[[TYPE_ID_EQ:.*]] = icmp eq i32 %[[EH_SELECTOR_PHI1]], %[[EH_TYPE_ID]]
// LLVM:   br i1 %[[TYPE_ID_EQ]], label %[[BEGIN_CATCH:.*]], label %[[RESUME:.*]]
// LLVM: [[BEGIN_CATCH]]:
// LLVM:   %[[EXN_OBJ_PHI2:.*]] = phi ptr [ %[[EXN_OBJ_PHI1:.*]], %[[DISPATCH:.*]] ]
// LLVM:   %[[EH_SELECTOR_PHI2:.*]] = phi i32 [ %[[EH_SELECTOR_PHI1:.*]], %[[DISPATCH:.*]] ]
// LLVM:   %[[TOKEN:.*]] = call ptr @__cxa_begin_catch(ptr %[[EXN_OBJ_PHI2]])
// LLVM:   br label %[[CATCH_BODY:.*]]
// LLVM: [[CATCH_BODY]]:
// LLVM:   %[[LOAD:.*]] = load { i32, i32 }, ptr %[[TOKEN]], align 4
// LLVM:   store { i32, i32 } %[[LOAD]], ptr {{.*}}, align 4
// LLVM:   br label %[[END_CATCH:.*]]
// LLVM: [[END_CATCH]]:
// LLVM:   call void @__cxa_end_catch()
// LLVM:   br label %[[END_DISPATCH:.*]]
// LLVM: [[END_DISPATCH]]:
// LLVM:   br label %[[END_TRY:.*]]
// LLVM: [[END_TRY]]:
// LLVM:   br label %[[TRY_CONT:.*]]
// LLVM: [[RESUME]]:
// LLVM:   %[[EXN_OBJ_PHI3:.*]] = phi ptr [ %[[EXN_OBJ_PHI1:.*]], %[[DISPATCH:.*]] ]
// LLVM:   %[[EH_SELECTOR_PHI3:.*]] = phi i32 [ %[[EH_SELECTOR_PHI1:.*]], %[[DISPATCH:.*]] ]
// LLVM:   %[[TMP_EXCEPTION_INFO:.*]] = insertvalue { ptr, i32 } poison, ptr %[[EXN_OBJ_PHI3]], 0
// LLVM:   %[[EXCEPTION_INFO:.*]] = insertvalue { ptr, i32 } %[[TMP_EXCEPTION_INFO]], i32 %[[EH_SELECTOR_PHI3]], 1
// LLVM:   resume { ptr, i32 } %[[EXCEPTION_INFO]]
// LLVM: [[TRY_CONT]]:
// LLVM:   br label %[[DONE:.*]]
// LLVM: [[DONE]]:
// LLVM:   ret void

// OGCG: define {{.*}} void @_Z58call_function_inside_try_catch_with_complex_exception_typev() {{.*}} personality ptr @__gxx_personality_v0
// OGCG:   %[[EXCEPTION_ADDR:.*]] = alloca ptr, align 8
// OGCG:   %[[EH_TYPE_ID_ADDR:.*]] = alloca i32, align 4
// OGCG:   %[[E_ADDR:.*]] = alloca { i32, i32 }, align 4
// OGCG:   %[[CALL:.*]] = invoke noundef i32 @_Z8divisionv()
// OGCG:           to label %[[INVOKE_NORMAL:.*]] unwind label %[[INVOKE_UNWIND:.*]]
// OGCG: [[INVOKE_NORMAL]]:
// OGCG:    br label %[[TRY_CONT:.*]]
// OGCG: [[INVOKE_UNWIND]]:
// OGCG:   %[[LANDING_PAD:.*]] = landingpad { ptr, i32 }
// OGCG:           catch ptr @_ZTICi
// OGCG:   %[[EXCEPTION:.*]] = extractvalue { ptr, i32 } %[[LANDING_PAD]], 0
// OGCG:   store ptr %[[EXCEPTION]], ptr %[[EXCEPTION_ADDR]], align 8
// OGCG:   %[[EH_TYPE_ID:.*]] = extractvalue { ptr, i32 } %[[LANDING_PAD]], 1
// OGCG:   store i32 %[[EH_TYPE_ID]], ptr %[[EH_TYPE_ID_ADDR]], align 4
// OGCG:   br label %[[CATCH_DISPATCH:.*]]
// OGCG: [[CATCH_DISPATCH]]:
// OGCG:   %[[TMP_EH_TYPE_ID:.*]] = load i32, ptr %[[EH_TYPE_ID_ADDR]], align 4
// OGCG:   %[[EH_TYPE_ID:.*]] = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTICi)
// OGCG:   %[[TYPE_ID_EQ:.*]] = icmp eq i32 %[[TMP_EH_TYPE_ID]], %[[EH_TYPE_ID]]
// OGCG:   br i1 %[[TYPE_ID_EQ]], label %[[CATCH_EXCEPTION:.*]], label %[[EH_RESUME:.*]]
// OGCG: [[CATCH_EXCEPTION]]:
// OGCG:   %[[TMP_EXCEPTION:.*]] = load ptr, ptr %[[EXCEPTION_ADDR]], align 8
// OGCG:   %[[BEGIN_CATCH:.*]] = call ptr @__cxa_begin_catch(ptr %[[TMP_EXCEPTION]])
// OGCG:   %[[EXCEPTION_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[BEGIN_CATCH]], i32 0, i32 0
// OGCG:   %[[EXCEPTION_REAL:.*]] = load i32, ptr %[[EXCEPTION_REAL_PTR]], align 4
// OGCG:   %[[EXCEPTION_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[BEGIN_CATCH]], i32 0, i32 1
// OGCG:   %[[EXCEPTION_IMAG:.*]] = load i32, ptr %[[EXCEPTION_IMAG_PTR]], align 4
// OGCG:   %[[E_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[E_ADDR]], i32 0, i32 0
// OGCG:   %[[E_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[E_ADDR]], i32 0, i32 1
// OGCG:   store i32 %[[EXCEPTION_REAL]], ptr %[[E_REAL_PTR]], align 4
// OGCG:   store i32 %[[EXCEPTION_IMAG]], ptr %[[E_IMAG_PTR]], align 4
// OGCG:   call void @__cxa_end_catch()
// OGCG:   br label %[[TRY_CONT]]
// OGCG: [[TRY_CONT]]:
// OGCG:   ret void
// OGCG: [[EH_RESUME]]:
// OGCG:   %[[TMP_EXCEPTION:.*]] = load ptr, ptr %[[EXCEPTION_ADDR]], align 8
// OGCG:   %[[TMP_EH_TYPE_ID:.*]] = load i32, ptr %[[EH_TYPE_ID_ADDR]], align 4
// OGCG:   %[[TMP_EXCEPTION_INFO:.*]] = insertvalue { ptr, i32 } poison, ptr %[[TMP_EXCEPTION]], 0
// OGCG:   %[[EXCEPTION_INFO:.*]] = insertvalue { ptr, i32 } %[[TMP_EXCEPTION_INFO]], i32 %[[TMP_EH_TYPE_ID]], 1
// OGCG:   resume { ptr, i32 } %[[EXCEPTION_INFO]]

void call_function_inside_try_catch_with_array_exception_type() {
  try {
    division();
  } catch (int e[]) {
  }
}

// CIR: cir.func {{.*}} @_Z56call_function_inside_try_catch_with_array_exception_typev() personality(@__gxx_personality_v0)
// CIR: cir.scope {
// CIR:   %[[E_ADDR:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["e"]
// CIR:   cir.try {
// CIR:     %[[CALL:.*]] = cir.call @_Z8divisionv()
// CIR:     cir.yield
// CIR:   } catch [type #cir.global_view<@_ZTIPi> : !cir.ptr<!u8i>] (%[[EH_TOKEN:.*]]: !cir.eh_token{{.*}}) {
// CIR:     %[[CATCH_TOKEN:.*]], %[[EXN_PTR:.*]] = cir.begin_catch %[[EH_TOKEN]] : !cir.eh_token -> (!cir.catch_token, !cir.ptr<!s32i>)
// CIR:     cir.cleanup.scope {
// CIR:       cir.store {{.*}} %[[EXN_PTR]], %[[E_ADDR]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR:       cir.yield
// CIR:     } cleanup {{.*}} {
// CIR:       cir.end_catch %[[CATCH_TOKEN]] : !cir.catch_token
// CIR:       cir.yield
// CIR:     }
// CIR:     cir.yield
// CIR:   } unwind (%{{.*}}: !cir.eh_token{{.*}}) {
// CIR:     cir.resume %{{.*}} : !cir.eh_token
// CIR:   }
// CIR: }

// LLVM: define {{.*}} void @_Z56call_function_inside_try_catch_with_array_exception_typev() {{.*}} personality ptr @__gxx_personality_v0
// LLVM:   br label %[[TRY_SCOPE:.*]]
// LLVM: [[TRY_SCOPE]]:
// LLVM:   br label %[[TRY_BEGIN:.*]]
// LLVM: [[TRY_BEGIN]]:
// LLVM:   %[[CALL:.*]] = invoke i32 @_Z8divisionv()
// LLVM:           to label %[[INVOKE_CONT:.*]] unwind label %[[LANDING_PAD:.*]]
// LLVM: [[INVOKE_CONT]]:
// LLVM:   br label %[[TRY_CONT:.*]]
// LLVM: [[LANDING_PAD]]:
// LLVM:   %[[LP:.*]] = landingpad { ptr, i32 }
// LLVM:           catch ptr @_ZTIPi
// LLVM:   %[[EXN_OBJ:.*]] = extractvalue { ptr, i32 } %[[LP]], 0
// LLVM:   %[[EH_SELECTOR_VAL:.*]] = extractvalue { ptr, i32 } %[[LP]], 1
// LLVM:   br label %[[CATCH:.*]]
// LLVM: [[CATCH]]:
// LLVM:   %[[EXN_OBJ_PHI:.*]] = phi ptr [ %[[EXN_OBJ:.*]], %[[LANDING_PAD:.*]] ]
// LLVM:   %[[EH_SELECTOR_PHI:.*]] = phi i32 [ %[[EH_SELECTOR_VAL:.*]], %[[LANDING_PAD:.*]] ]
// LLVM:   br label %[[DISPATCH:.*]]
// LLVM: [[DISPATCH]]:
// LLVM:   %[[EXN_OBJ_PHI1:.*]] = phi ptr [ %[[EXN_OBJ_PHI:.*]], %[[CATCH:.*]] ]
// LLVM:   %[[EH_SELECTOR_PHI1:.*]] = phi i32 [ %[[EH_SELECTOR_PHI:.*]], %[[CATCH:.*]] ]
// LLVM:   %[[EH_TYPE_ID:.*]] = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIPi)
// LLVM:   %[[TYPE_ID_EQ:.*]] = icmp eq i32 %[[EH_SELECTOR_PHI1]], %[[EH_TYPE_ID]]
// LLVM:   br i1 %[[TYPE_ID_EQ]], label %[[BEGIN_CATCH:.*]], label %[[RESUME:.*]]
// LLVM: [[BEGIN_CATCH]]:
// LLVM:   %[[EXN_OBJ_PHI2:.*]] = phi ptr [ %[[EXN_OBJ_PHI1:.*]], %[[DISPATCH:.*]] ]
// LLVM:   %[[EH_SELECTOR_PHI2:.*]] = phi i32 [ %[[EH_SELECTOR_PHI1:.*]], %[[DISPATCH:.*]] ]
// LLVM:   %[[TOKEN:.*]] = call ptr @__cxa_begin_catch(ptr %[[EXN_OBJ_PHI2]])
// LLVM:   br label %[[CATCH_BODY:.*]]
// LLVM: [[CATCH_BODY]]:
// LLVM:   store ptr %[[TOKEN]], ptr {{.*}}, align 8
// LLVM:   br label %[[END_CATCH:.*]]
// LLVM: [[END_CATCH]]:
// LLVM:   call void @__cxa_end_catch()
// LLVM:   br label %[[END_DISPATCH:.*]]
// LLVM: [[END_DISPATCH]]:
// LLVM:   br label %[[END_TRY:.*]]
// LLVM: [[END_TRY]]:
// LLVM:   br label %[[TRY_CONT:.*]]
// LLVM: [[RESUME]]:
// LLVM:   %[[EXN_OBJ_PHI3:.*]] = phi ptr [ %[[EXN_OBJ_PHI1:.*]], %[[DISPATCH:.*]] ]
// LLVM:   %[[EH_SELECTOR_PHI3:.*]] = phi i32 [ %[[EH_SELECTOR_PHI1:.*]], %[[DISPATCH:.*]] ]
// LLVM:   %[[TMP_EXCEPTION_INFO:.*]] = insertvalue { ptr, i32 } poison, ptr %[[EXN_OBJ_PHI3]], 0
// LLVM:   %[[EXCEPTION_INFO:.*]] = insertvalue { ptr, i32 } %[[TMP_EXCEPTION_INFO]], i32 %[[EH_SELECTOR_PHI3]], 1
// LLVM:   resume { ptr, i32 } %[[EXCEPTION_INFO]]
// LLVM: [[TRY_CONT]]:
// LLVM:   br label %[[DONE:.*]]
// LLVM: [[DONE]]:
// LLVM:   ret void

// OGCG: define {{.*}} void @_Z56call_function_inside_try_catch_with_array_exception_typev() {{.*}} personality ptr @__gxx_personality_v0
// OGCG:   %[[EXCEPTION_ADDR:.*]] = alloca ptr, align 8
// OGCG:   %[[EH_TYPE_ID_ADDR:.*]] = alloca i32, align 4
// OGCG:   %[[E_ADDR:.*]] = alloca ptr, align 8
// OGCG:   %[[CALL:.*]] = invoke noundef i32 @_Z8divisionv()
// OGCG:          to label %[[INVOKE_NORMAL:.*]] unwind label %[[INVOKE_UNWIND:.*]]
// OGCG: [[INVOKE_NORMAL]]:
// OGCG:   br label %[[TRY_CONT:.*]]
// OGCG: [[INVOKE_UNWIND]]:
// OGCG:   %[[LANDING_PAD:.*]] = landingpad { ptr, i32 }
// OGCG:           catch ptr @_ZTIPi
// OGCG:   %[[EXCEPTION:.*]] = extractvalue { ptr, i32 } %[[LANDING_PAD]], 0
// OGCG:   store ptr %[[EXCEPTION]], ptr %[[EXCEPTION_ADDR]], align 8
// OGCG:   %[[EH_TYPE_ID:.*]] = extractvalue { ptr, i32 } %[[LANDING_PAD]], 1
// OGCG:   store i32 %[[EH_TYPE_ID]], ptr %[[EH_TYPE_ID_ADDR]], align 4
// OGCG:   br label %[[CATCH_DISPATCH:.*]]
// OGCG: [[CATCH_DISPATCH]]:
// OGCG:   %[[TMP_EH_TYPE_ID:.*]] = load i32, ptr %ehselector.slot, align 4
// OGCG:   %[[EH_TYPE_ID:.*]] = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIPi)
// OGCG:   %[[TYPE_ID_EQ:.*]] = icmp eq i32 %[[TMP_EH_TYPE_ID]], %[[EH_TYPE_ID]]
// OGCG:   br i1 %[[TYPE_ID_EQ]], label %[[CATCH_EXCEPTION:.*]], label %[[EH_RESUME:.*]]
// OGCG: [[CATCH_EXCEPTION]]:
// OGCG:   %[[TMP_EXCEPTION:.*]] = load ptr, ptr %[[EXCEPTION_ADDR]], align 8
// OGCG:   %[[BEGIN_CATCH:.*]] = call ptr @__cxa_begin_catch(ptr %[[TMP_EXCEPTION]])
// OGCG:   store ptr %[[BEGIN_CATCH]], ptr %[[E_ADDR]], align 8
// OGCG:   call void @__cxa_end_catch()
// OGCG:   br label %[[TRY_CONT]]
// OGCG: [[TRY_CONT]]:
// OGCG:   ret void
// OGCG: [[EH_RESUME]]:
// OGCG:   %[[TMP_EXCEPTION:.*]] = load ptr, ptr %[[EXCEPTION_ADDR]], align 8
// OGCG:   %[[TMP_EH_TYPE_ID:.*]] = load i32, ptr %[[EH_TYPE_ID_ADDR]], align 4
// OGCG:   %[[TMP_EXCEPTION_INFO:.*]] = insertvalue { ptr, i32 } poison, ptr %[[TMP_EXCEPTION]], 0
// OGCG:   %[[EXCEPTION_INFO:.*]] = insertvalue { ptr, i32 } %[[TMP_EXCEPTION_INFO]], i32 %[[TMP_EH_TYPE_ID]], 1
// OGCG:   resume { ptr, i32 } %[[EXCEPTION_INFO]]

void call_function_inside_try_catch_with_exception_type_and_catch_all() {
  try {
    division();
  } catch (int e) {
  } catch (...) {
  }
}

// CIR: cir.func {{.*}} @_Z64call_function_inside_try_catch_with_exception_type_and_catch_allv() personality(@__gxx_personality_v0)
// CIR: cir.scope {
// CIR:   %[[EXCEPTION_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["e"]
// CIR:   cir.try {
// CIR:     %[[CALL:.*]] = cir.call @_Z8divisionv()
// CIR:     cir.yield
// CIR:   } catch [type #cir.global_view<@_ZTIi> : !cir.ptr<!u8i>] (%[[EH_TOKEN:.*]]: !cir.eh_token{{.*}}) {
// CIR:     %[[CATCH_TOKEN:.*]], %[[EXN_PTR:.*]] = cir.begin_catch %[[EH_TOKEN]] : !cir.eh_token -> (!cir.catch_token, !cir.ptr<!s32i>)
// CIR:     cir.cleanup.scope {
// CIR:       %[[TMP:.*]] = cir.load {{.*}} %[[EXN_PTR]] : !cir.ptr<!s32i>, !s32i
// CIR:       cir.store {{.*}} %[[TMP]], %[[EXCEPTION_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:       cir.yield
// CIR:     } cleanup {{.*}} {
// CIR:       cir.end_catch %[[CATCH_TOKEN]] : !cir.catch_token
// CIR:       cir.yield
// CIR:     }
// CIR:     cir.yield
// CIR:   } catch all (%[[EH_TOKEN2:.*]]: !cir.eh_token{{.*}}) {
// CIR:     %[[CATCH_TOKEN2:.*]], %{{.*}} = cir.begin_catch %[[EH_TOKEN2]] : !cir.eh_token -> (!cir.catch_token, !cir.ptr<!void>)
// CIR:     cir.cleanup.scope {
// CIR:       cir.yield
// CIR:     } cleanup {{.*}} {
// CIR:       cir.end_catch %[[CATCH_TOKEN2]] : !cir.catch_token
// CIR:       cir.yield
// CIR:     }
// CIR:     cir.yield
// CIR:   }
// CIR: }

// LLVM: define {{.*}} void @_Z64call_function_inside_try_catch_with_exception_type_and_catch_allv() {{.*}} personality ptr @__gxx_personality_v0
// LLVM:   br label %[[TRY_SCOPE:.*]]
// LLVM: [[TRY_SCOPE]]:
// LLVM:   br label %[[TRY_BEGIN:.*]]
// LLVM: [[TRY_BEGIN]]:
// LLVM:   %[[CALL:.*]] = invoke i32 @_Z8divisionv()
// LLVM:           to label %[[INVOKE_CONT:.*]] unwind label %[[LANDING_PAD:.*]]
// LLVM: [[INVOKE_CONT]]:
// LLVM:   br label %[[TRY_CONT:.*]]
// LLVM: [[LANDING_PAD]]:
// LLVM:   %[[LP:.*]] = landingpad { ptr, i32 }
// LLVM:           catch ptr @_ZTIi
// LLVM:   %[[EXN_OBJ:.*]] = extractvalue { ptr, i32 } %[[LP]], 0
// LLVM:   %[[EH_SELECTOR_VAL:.*]] = extractvalue { ptr, i32 } %[[LP]], 1
// LLVM:   br label %[[CATCH:.*]]
// LLVM: [[CATCH]]:
// LLVM:   %[[EXN_OBJ_PHI:.*]] = phi ptr [ %[[EXN_OBJ:.*]], %[[LANDING_PAD:.*]] ]
// LLVM:   %[[EH_SELECTOR_PHI:.*]] = phi i32 [ %[[EH_SELECTOR_VAL:.*]], %[[LANDING_PAD:.*]] ]
// LLVM:   br label %[[DISPATCH:.*]]
// LLVM: [[DISPATCH]]:
// LLVM:   %[[EXN_OBJ_PHI1:.*]] = phi ptr [ %[[EXN_OBJ_PHI:.*]], %[[CATCH:.*]] ]
// LLVM:   %[[EH_SELECTOR_PHI1:.*]] = phi i32 [ %[[EH_SELECTOR_PHI:.*]], %[[CATCH:.*]] ]
// LLVM:   %[[EH_TYPE_ID:.*]] = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
// LLVM:   %[[TYPE_ID_EQ:.*]] = icmp eq i32 %[[EH_SELECTOR_PHI1]], %[[EH_TYPE_ID]]
// LLVM:   br i1 %[[TYPE_ID_EQ]], label %[[BEGIN_CATCH:.*]], label %[[CATCH_ALL:.*]]
// LLVM: [[BEGIN_CATCH]]:
// LLVM:   %[[EXN_OBJ_PHI2:.*]] = phi ptr [ %[[EXN_OBJ_PHI1:.*]], %[[DISPATCH:.*]] ]
// LLVM:   %[[EH_SELECTOR_PHI2:.*]] = phi i32 [ %[[EH_SELECTOR_PHI1:.*]], %[[DISPATCH:.*]] ]
// LLVM:   %[[TOKEN:.*]] = call ptr @__cxa_begin_catch(ptr %[[EXN_OBJ_PHI2]])
// LLVM:   br label %[[CATCH_BODY:.*]]
// LLVM: [[CATCH_BODY]]:
// LLVM:   %[[LOAD:.*]] = load i32, ptr %[[TOKEN]], align 4
// LLVM:   store i32 %[[LOAD]], ptr {{.*}}, align 4
// LLVM:   br label %[[END_CATCH:.*]]
// LLVM: [[END_CATCH]]:
// LLVM:   call void @__cxa_end_catch()
// LLVM:   br label %[[END_DISPATCH:.*]]
// LLVM: [[END_DISPATCH]]:
// LLVM:   br label %[[END_TRY:.*]]
// LLVM: [[END_TRY]]:
// LLVM:   br label %[[TRY_CONT:.*]]
// LLVM: [[CATCH_ALL]]:
// LLVM:   %[[EXN_OBJ_PHI3:.*]] = phi ptr [ %[[EXN_OBJ_PHI1:.*]], %[[DISPATCH:.*]] ]
// LLVM:   %[[EH_SELECTOR_PHI3:.*]] = phi i32 [ %[[EH_SELECTOR_PHI1:.*]], %[[DISPATCH:.*]] ]
// LLVM:   %[[TOKEN2:.*]] = call ptr @__cxa_begin_catch(ptr %[[EXN_OBJ_PHI3]])
// LLVM:   br label %[[CATCH_ALL_BODY:.*]]
// LLVM: [[CATCH_ALL_BODY]]:
// LLVM:   br label %[[END_CATCH2:.*]]
// LLVM: [[END_CATCH2]]:
// LLVM:   call void @__cxa_end_catch()
// LLVM:   br label %[[END_DISPATCH2:.*]]
// LLVM: [[END_DISPATCH2]]:
// LLVM:   br label %[[END_TRY2:.*]]
// LLVM: [[END_TRY2]]:
// LLVM:   br label %[[TRY_CONT:.*]]
// LLVM: [[TRY_CONT]]:
// LLVM:   br label %[[DONE:.*]]
// LLVM: [[DONE]]:
// LLVM:   ret void

// OGCG: define {{.*}} void @_Z64call_function_inside_try_catch_with_exception_type_and_catch_allv() {{.*}} personality ptr @__gxx_personality_v0
// OGCG:   %[[EXCEPTION_ADDR:.*]] = alloca ptr, align 8
// OGCG:   %[[EH_TYPE_ID_ADDR:.*]] = alloca i32, align 4
// OGCG:   %[[E_ADDR:.*]] = alloca i32, align 4
// OGCG:   %[[CALL:.*]] = invoke noundef i32 @_Z8divisionv()
// OGCG:           to label %[[INVOKE_NORMAL:.*]] unwind label %[[INVOKE_UNWIND:.*]]
// OGCG: [[INVOKE_NORMAL]]:
// OGCG:   br label %try.cont
// OGCG: [[INVOKE_UNWIND]]:
// OGCG:   %[[LANDING_PAD:.*]] = landingpad { ptr, i32 }
// OGCG:           catch ptr @_ZTIi
// OGCG:   %[[EXCEPTION:.*]] = extractvalue { ptr, i32 } %[[LANDING_PAD]], 0
// OGCG:   store ptr %[[EXCEPTION]], ptr %[[EXCEPTION_ADDR]], align 8
// OGCG:   %[[EH_TYPE_ID:.*]] = extractvalue { ptr, i32 } %[[LANDING_PAD]], 1
// OGCG:   store i32 %[[EH_TYPE_ID]], ptr %[[EH_TYPE_ID_ADDR]], align 4
// OGCG:   br label %[[CATCH_DISPATCH:.*]]
// OGCG: [[CATCH_DISPATCH]]:
// OGCG:   %[[TMP_EH_TYPE_ID:.*]] = load i32, ptr %[[EH_TYPE_ID_ADDR]], align 4
// OGCG:   %[[EH_TYPE_ID:.*]] = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
// OGCG:   %[[TYPE_ID_EQ:.*]] = icmp eq i32 %[[TMP_EH_TYPE_ID]], %[[EH_TYPE_ID]]
// OGCG:   br i1 %[[TYPE_ID_EQ]], label %[[CATCH_EXCEPTION:.*]], label %[[CATCH_ALL:.*]]
// OGCG: [[CATCH_EXCEPTION]]:
// OGCG:   %[[TMP_EXCEPTION:.*]] = load ptr, ptr %[[EXCEPTION_ADDR]], align 8
// OGCG:   %[[BEGIN_CATCH:.*]] = call ptr @__cxa_begin_catch(ptr %[[TMP_EXCEPTION]])
// OGCG:   %[[TMP_BEGIN_CATCH:.*]] = load i32, ptr %[[BEGIN_CATCH]], align 4
// OGCG:   store i32 %[[TMP_BEGIN_CATCH]], ptr %[[E_ADDR]], align 4
// OGCG:   call void @__cxa_end_catch()
// OGCG:   br label %[[TRY_CONT:.*]]
// OGCG: [[TRY_CONT]]:
// OGCG:   ret void
// OGCG: [[CATCH_ALL]]:
// OGCG:   %[[TMP_EXCEPTION:.*]] = load ptr, ptr %[[EXCEPTION_ADDR]], align 8
// OGCG:   %[[BEGIN_CATCH:.*]] = call ptr @__cxa_begin_catch(ptr %[[TMP_EXCEPTION]])
// OGCG:   call void @__cxa_end_catch()
// OGCG:   br label %[[TRY_CONT]]

struct S {
  ~S();
};

void cleanup_inside_try_body() {
  try {
    S s;
    division();
  } catch (...) {
  }
}

// CIR: cir.func {{.*}} @_Z23cleanup_inside_try_bodyv(){{.*}} personality(@__gxx_personality_v0) {
// CIR: cir.scope {
// CIR:   %[[S:.*]] = cir.alloca !rec_S, !cir.ptr<!rec_S>, ["s"]
// CIR:   cir.try {
// CIR:     cir.cleanup.scope {
// CIR:       cir.call @_Z8divisionv()
// CIR:       cir.yield
// CIR:     } cleanup  all {
// CIR:       cir.call @_ZN1SD1Ev(%[[S]])
// CIR:       cir.yield
// CIR:     }
// CIR:     cir.yield
// CIR:   } catch all (%[[TOKEN:.*]]: !cir.eh_token {{.*}}) {
// CIR:     %[[CATCH_TOKEN:.*]], %[[EXN_PTR:.*]] = cir.begin_catch %[[TOKEN]] : !cir.eh_token -> (!cir.catch_token, !cir.ptr<!void>)
// CIR:     cir.cleanup.scope {
// CIR:       cir.yield
// CIR:     } cleanup  all {
// CIR:       cir.end_catch %[[CATCH_TOKEN]] : !cir.catch_token
// CIR:       cir.yield
// CIR:     }
// CIR:     cir.yield
// CIR:   }
// CIR: }

// LLVM: define {{.*}} void @_Z23cleanup_inside_try_bodyv() {{.*}} personality ptr @__gxx_personality_v0
// LLVM:   br label %[[TRY_SCOPE:.*]]
// LLVM: [[TRY_SCOPE]]:
// LLVM:   br label %[[TRY_BEGIN:.*]]
// LLVM: [[TRY_BEGIN]]:
// LLVM:   br label %[[CLEANUP_SCOPE:.*]]
// LLVM: [[CLEANUP_SCOPE]]:
// LLVM:   %[[CALL:.*]] = invoke i32 @_Z8divisionv()
// LLVM:           to label %[[INVOKE_CONT:.*]] unwind label %[[LANDING_PAD:.*]]
// LLVM: [[INVOKE_CONT]]:
// LLVM:   br label %[[CLEANUP:.*]]
// LLVM: [[CLEANUP]]:
// LLVM:   call void @_ZN1SD1Ev(ptr {{.*}})
// LLVM:   br label %[[END_CLEANUP:.*]]
// LLVM: [[END_CLEANUP]]:
// LLVM:   br label %[[TRY_CONT:.*]]
// LLVM: [[LANDING_PAD]]:
// LLVM:   %[[LP:.*]] = landingpad { ptr, i32 }
// LLVM:           cleanup
// LLVM:   %[[EXN_OBJ:.*]] = extractvalue { ptr, i32 } %[[LP]], 0
// LLVM:   %[[EH_SELECTOR_VAL:.*]] = extractvalue { ptr, i32 } %[[LP]], 1
// LLVM:   br label %[[CLEANUP_LANDING:.*]]
// LLVM: [[CLEANUP_LANDING]]:
// LLVM:   %[[EXN_OBJ_PHI:.*]] = phi ptr [ %[[EXN_OBJ:.*]], %[[LANDING_PAD:.*]] ]
// LLVM:   %[[EH_SELECTOR_PHI:.*]] = phi i32 [ %[[EH_SELECTOR_VAL:.*]], %[[LANDING_PAD:.*]] ]
// LLVM:   call void @_ZN1SD1Ev(ptr {{.*}})
// LLVM:   br label %[[CATCH:.*]]
// LLVM: [[CATCH]]:
// LLVM:   %[[EXN_OBJ_PHI1:.*]] = phi ptr [ %[[EXN_OBJ_PHI:.*]], %[[CLEANUP_LANDING:.*]] ]
// LLVM:   %[[EH_SELECTOR_PHI1:.*]] = phi i32 [ %[[EH_SELECTOR_PHI:.*]], %[[CLEANUP_LANDING:.*]] ]
// LLVM:   br label %[[BEGIN_CATCH:.*]]
// LLVM: [[BEGIN_CATCH]]:
// LLVM:   %[[EXN_OBJ_PHI2:.*]] = phi ptr [ %[[EXN_OBJ_PHI1:.*]], %[[CATCH:.*]] ]
// LLVM:   %[[EH_SELECTOR_PHI2:.*]] = phi i32 [ %[[EH_SELECTOR_PHI1:.*]], %[[CATCH:.*]] ]
// LLVM:   %[[TOKEN:.*]] = call ptr @__cxa_begin_catch(ptr %[[EXN_OBJ_PHI2]])
// LLVM:   br label %[[CATCH_BODY:.*]]
// LLVM: [[CATCH_BODY]]:
// LLVM:   br label %[[END_CATCH:.*]]
// LLVM: [[END_CATCH]]:
// LLVM:   call void @__cxa_end_catch()
// LLVM:   br label %[[END_DISPATCH:.*]]
// LLVM: [[END_DISPATCH]]:
// LLVM:   br label %[[END_TRY:.*]]
// LLVM: [[END_TRY]]:
// LLVM:   br label %[[TRY_CONT:.*]]
// LLVM: [[TRY_CONT]]:
// LLVM:   br label %[[DONE:.*]]
// LLVM: [[DONE]]:
// LLVM:   ret void

// OGCG: define {{.*}} void @_Z23cleanup_inside_try_bodyv() {{.*}} personality ptr @__gxx_personality_v0 {
// OGCG:   %[[S:.*]] = alloca %struct.S
// OGCG:   %[[EXN_SLOT:.*]] = alloca ptr
// OGCG:   %[[EHSELECTOR_SLOT:.*]] = alloca i32
// OGCG:   %[[CALL:.*]] = invoke {{.*}} i32 @_Z8divisionv()
// OGCG:           to label %[[INVOKE_CONT:.*]] unwind label %[[LANDING_PAD:.*]]
// OGCG: [[INVOKE_CONT]]:
// OGCG:   call void @_ZN1SD1Ev(ptr noundef nonnull align 1 dereferenceable(1) %[[S]])
// OGCG:   br label %[[TRY_CONT:.*]]
// OGCG: [[LANDING_PAD]]:
// OGCG:   %[[LANDING_PAD:.*]] = landingpad { ptr, i32 }
// OGCG:           catch ptr null
// OGCG:   %[[EXCEPTION:.*]] = extractvalue { ptr, i32 } %[[LANDING_PAD]], 0
// OGCG:   store ptr %[[EXCEPTION]], ptr %[[EXN_SLOT]]
// OGCG:   %[[EH_TYPE_ID:.*]] = extractvalue { ptr, i32 } %[[LANDING_PAD]], 1
// OGCG:   store i32 %[[EH_TYPE_ID]], ptr %[[EHSELECTOR_SLOT]]
// OGCG:   call void @_ZN1SD1Ev(ptr noundef nonnull align 1 dereferenceable(1) %[[S]])
// OGCG:   br label %[[CATCH:.*]]
// OGCG: [[CATCH]]:
// OGCG:   %[[EXCEPTION:.*]] = load ptr, ptr %[[EXN_SLOT]]
// OGCG:   %[[BEGIN_CATCH:.*]] = call ptr @__cxa_begin_catch(ptr %[[EXCEPTION]])
// OGCG:   call void @__cxa_end_catch()
// OGCG:   br label %[[TRY_CONT]]

struct Record {
  int x;
  int y;
};

void call_function_inside_try_catch_with_ref_ptr_of_record_exception_type() {
  try {
    division();
  } catch (Record *&ref_ptr) {
  }
}

// CIR: cir.func {{.*}} @_Z68call_function_inside_try_catch_with_ref_ptr_of_record_exception_typev(){{.*}} personality(@__gxx_personality_v0) {
// CIR:   %[[E_ADDR:.*]] = cir.alloca !cir.ptr<!cir.ptr<!rec_Record>>, !cir.ptr<!cir.ptr<!cir.ptr<!rec_Record>>>, ["ref_ptr", const]
// CIR:   %[[EXN_BYREF_TMP:.*]] = cir.alloca !cir.ptr<!rec_Record>, !cir.ptr<!cir.ptr<!rec_Record>>, ["exn.byref.tmp"]
// CIR:   cir.try {
// CIR:     %[[CALL:.*]] = cir.call @_Z8divisionv() : () -> (!s32i {llvm.noundef})
// CIR:     cir.yield
// CIR:   } catch [type #cir.global_view<@_ZTIP6Record> : !cir.ptr<!u8i>] (%[[EH_TOKEN:.*]]: !cir.eh_token {{.*}}) {
// CIR:     %[[CATCH_TOKEN:.*]], %[[EXN_PTR:.*]] = cir.begin_catch %[[EH_TOKEN]] : !cir.eh_token -> (!cir.catch_token, !cir.ptr<!cir.ptr<!rec_Record>>)
// CIR:     cir.cleanup.scope {
// CIR:       %[[EXN_PTR_REC_PTR:.*]] = cir.cast bitcast %[[EXN_PTR]] : !cir.ptr<!cir.ptr<!rec_Record>> -> !cir.ptr<!rec_Record>
// CIR:       cir.store {{.*}} %[[EXN_PTR_REC_PTR]], %[[EXN_BYREF_TMP]] : !cir.ptr<!rec_Record>, !cir.ptr<!cir.ptr<!rec_Record>>
// CIR:       cir.store {{.*}} %[[EXN_BYREF_TMP]], %[[E_ADDR]] : !cir.ptr<!cir.ptr<!rec_Record>>, !cir.ptr<!cir.ptr<!cir.ptr<!rec_Record>>>
// CIR:       cir.yield
// CIR:     } cleanup all {
// CIR:       cir.end_catch %[[CATCH_TOKEN]] : !cir.catch_token
// CIR:       cir.yield
// CIR:     }
// CIR:     cir.yield
// CIR:   } unwind (%{{.*}}: !cir.eh_token {{.*}}) {
// CIR:     cir.resume %{{.*}} : !cir.eh_token
// CIR:   }
// CIR: }

// LLVM: define {{.*}} void @_Z68call_function_inside_try_catch_with_ref_ptr_of_record_exception_typev() {{.*}} personality ptr @__gxx_personality_v0
// LLVM:   %[[E_ADDR:.*]] = alloca ptr
// LLVM:   %[[EXN_BYREF_TMP:.*]] = alloca ptr
// LLVM:   br label %[[TRY_SCOPE:.*]]
// LLVM: [[TRY_SCOPE]]:
// LLVM:   br label %[[TRY_BEGIN:.*]]
// LLVM: [[TRY_BEGIN]]:
// LLVM:   invoke i32 @_Z8divisionv()
// LLVM:           to label %[[INVOKE_CONT:.*]] unwind label %[[LANDING_PAD:.*]]
// LLVM: [[INVOKE_CONT:.*]]:
// LLVM:   br label %[[TRY_CONT:.*]]
// LLVM: [[LANDING_PAD]]:
// LLVM:   %[[LP:.*]] = landingpad { ptr, i32 }
// LLVM:           catch ptr @_ZTIP6Record
// LLVM:   %[[EXN_OBJ:.*]] = extractvalue { ptr, i32 } %[[LP]], 0
// LLVM:   %[[EH_SELECTOR_VAL:.*]] = extractvalue { ptr, i32 } %[[LP]], 1
// LLVM:   br label %[[CATCH:.*]]
// LLVM: [[CATCH]]:
// LLVM:   %[[EXN_OBJ_PHI:.*]] = phi ptr [ %[[EXN_OBJ:.*]], %[[LANDING_PAD:.*]] ]
// LLVM:   %[[EH_SELECTOR_PHI:.*]] = phi i32 [ %[[EH_SELECTOR_VAL:.*]], %[[LANDING_PAD:.*]] ]
// LLVM:   br label %[[DISPATCH:.*]]
// LLVM: [[DISPATCH]]:
// LLVM:   %[[EXN_OBJ_PHI1:.*]] = phi ptr [ %[[EXN_OBJ_PHI:.*]], %[[CATCH:.*]] ]
// LLVM:   %[[EH_SELECTOR_PHI1:.*]] = phi i32 [ %[[EH_SELECTOR_PHI:.*]], %[[CATCH:.*]] ]
// LLVM:   %[[EH_TYPE_ID:.*]] = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIP6Record)
// LLVM:   %[[TYPE_ID_EQ:.*]] = icmp eq i32 %[[EH_SELECTOR_PHI1]], %[[EH_TYPE_ID]]
// LLVM:   br i1 %[[TYPE_ID_EQ]], label %[[BEGIN_CATCH:.*]], label %[[RESUME:.*]]
// LLVM: [[BEGIN_CATCH]]:
// LLVM:   %[[EXN_OBJ_PHI2:.*]] = phi ptr [ %[[EXN_OBJ_PHI1:.*]], %[[DISPATCH:.*]] ]
// LLVM:   %[[EH_SELECTOR_PHI2:.*]] = phi i32 [ %[[EH_SELECTOR_PHI1:.*]], %[[DISPATCH:.*]] ]
// LLVM:   %[[TOKEN:.*]] = call ptr @__cxa_begin_catch(ptr %[[EXN_OBJ_PHI2]])
// LLVM:   br label %[[CATCH_BODY:.*]]
// LLVM: [[CATCH_BODY]]:
// LLVM:   store ptr %[[TOKEN]], ptr %[[EXN_BYREF_TMP]], align 8
// LLVM:   store ptr %[[EXN_BYREF_TMP]], ptr %[[E_ADDR]], align 8
// LLVM:   br label %[[END_CATCH:.*]]
// LLVM: [[END_CATCH]]:
// LLVM:   call void @__cxa_end_catch()
// LLVM:   br label %[[END_DISPATCH:.*]]
// LLVM: [[END_DISPATCH]]:
// LLVM:   br label %[[END_TRY:.*]]
// LLVM: [[END_TRY]]:
// LLVM:   br label %[[TRY_CONT:.*]]
// LLVM: [[RESUME]]:
// LLVM:   %[[EXN_OBJ_PHI3:.*]] = phi ptr [ %[[EXN_OBJ_PHI1:.*]], %[[DISPATCH:.*]] ]
// LLVM:   %[[EH_SELECTOR_PHI3:.*]] = phi i32 [ %[[EH_SELECTOR_PHI1:.*]], %[[DISPATCH:.*]] ]
// LLVM:   %[[TMP_EXCEPTION_INFO:.*]] = insertvalue { ptr, i32 } poison, ptr %[[EXN_OBJ_PHI3]], 0
// LLVM:   %[[EXCEPTION_INFO:.*]] = insertvalue { ptr, i32 } %[[TMP_EXCEPTION_INFO]], i32 %[[EH_SELECTOR_PHI3]], 1
// LLVM:   resume { ptr, i32 } %[[EXCEPTION_INFO]]
// LLVM: [[TRY_CONT]]:
// LLVM:   br label %[[DONE:.*]]
// LLVM: [[DONE]]:
// LLVM:   ret void

// OGCG: define {{.*}} void @_Z68call_function_inside_try_catch_with_ref_ptr_of_record_exception_typev() {{.*}} personality ptr @__gxx_personality_v0
// OGCG:   %[[EXCEPTION_ADDR:.*]] = alloca ptr, align 8
// OGCG:   %[[EH_TYPE_ID_ADDR:.*]] = alloca i32, align 4
// OGCG:   %[[E_ADDR:.*]] = alloca ptr, align 8
// OGCG:   %[[EXN_BYREF_TMP:.*]] = alloca ptr, align 8
// OGCG:   %[[CALL:.*]] = invoke noundef i32 @_Z8divisionv()
// OGCG:          to label %[[INVOKE_NORMAL:.*]] unwind label %[[INVOKE_UNWIND:.*]]
// OGCG: [[INVOKE_NORMAL]]:
// OGCG:   br label %[[TRY_CONT:.*]]
// OGCG: [[INVOKE_UNWIND]]:
// OGCG:   %[[LANDING_PAD:.*]] = landingpad { ptr, i32 }
// OGCG:           catch ptr @_ZTIP6Record
// OGCG:   %[[EXCEPTION:.*]] = extractvalue { ptr, i32 } %[[LANDING_PAD]], 0
// OGCG:   store ptr %[[EXCEPTION]], ptr %[[EXCEPTION_ADDR]], align 8
// OGCG:   %[[EH_TYPE_ID:.*]] = extractvalue { ptr, i32 } %[[LANDING_PAD]], 1
// OGCG:   store i32 %[[EH_TYPE_ID]], ptr %[[EH_TYPE_ID_ADDR]], align 4
// OGCG:   br label %[[CATCH_DISPATCH:.*]]
// OGCG: [[CATCH_DISPATCH]]:
// OGCG:   %[[TMP_EH_TYPE_ID:.*]] = load i32, ptr %ehselector.slot, align 4
// OGCG:   %[[EH_TYPE_ID:.*]] = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIP6Record)
// OGCG:   %[[TYPE_ID_EQ:.*]] = icmp eq i32 %[[TMP_EH_TYPE_ID]], %[[EH_TYPE_ID]]
// OGCG:   br i1 %[[TYPE_ID_EQ]], label %[[CATCH_EXCEPTION:.*]], label %[[EH_RESUME:.*]]
// OGCG: [[CATCH_EXCEPTION]]:
// OGCG:   %[[TMP_EXCEPTION:.*]] = load ptr, ptr %[[EXCEPTION_ADDR]], align 8
// OGCG:   %[[BEGIN_CATCH:.*]] = call ptr @__cxa_begin_catch(ptr %[[TMP_EXCEPTION]])
// OGCG:   store ptr %[[BEGIN_CATCH]], ptr %[[EXN_BYREF_TMP]], align 8
// OGCG:   store ptr %[[EXN_BYREF_TMP]], ptr %[[E_ADDR]], align 8
// OGCG:   call void @__cxa_end_catch()
// OGCG:   br label %[[TRY_CONT]]
// OGCG: [[TRY_CONT]]:
// OGCG:   ret void
// OGCG: [[EH_RESUME]]:
// OGCG:   %[[TMP_EXCEPTION:.*]] = load ptr, ptr %[[EXCEPTION_ADDR]], align 8
// OGCG:   %[[TMP_EH_TYPE_ID:.*]] = load i32, ptr %[[EH_TYPE_ID_ADDR]], align 4
// OGCG:   %[[TMP_EXCEPTION_INFO:.*]] = insertvalue { ptr, i32 } poison, ptr %[[TMP_EXCEPTION]], 0
// OGCG:   %[[EXCEPTION_INFO:.*]] = insertvalue { ptr, i32 } %[[TMP_EXCEPTION_INFO]], i32 %[[TMP_EH_TYPE_ID]], 1
// OGCG:   resume { ptr, i32 } %[[EXCEPTION_INFO]]
