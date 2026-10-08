// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefixes=LLVM,LLVMCIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefixes=LLVM,OGCG

void empty_try_block_with_catch_all() {
  try {} catch (...) {}
}

// CIR: cir.func{{.*}} @_Z30empty_try_block_with_catch_allv()
// CIR:   cir.return

// LLVM: define{{.*}} void @_Z30empty_try_block_with_catch_allv()
// LLVM:   ret void

void empty_try_block_with_catch_with_int_exception() {
  try {} catch (int e) {}
}

// CIR: cir.func{{.*}} @_Z45empty_try_block_with_catch_with_int_exceptionv()
// CIR:   cir.return

// LLVM: define{{.*}} void @_Z45empty_try_block_with_catch_with_int_exceptionv()
// LLVM:   ret void

void try_catch_with_empty_catch_all() {
  int a = 1;
  try {
    return;
    ++a;
  } catch (...) {
  }
}

// CIR: cir.func {{.*}} @_Z30try_catch_with_empty_catch_allv() personality(@__gxx_personality_v0)
// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} init : !cir.ptr<!s32i>
// CIR: %[[CONST_1:.*]] = cir.const #cir.int<1> : !s32i
// CIR: cir.store{{.*}} %[[CONST_1]], %[[A_ADDR]] : !s32i, !cir.ptr<!s32i
// CIR: cir.scope {
// CIR:   cir.try {
// CIR:     cir.return
// CIR:   ^bb1:  // no predecessors
// CIR:     %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:     %[[RESULT:.*]] = cir.inc nsw %[[TMP_A]] : !s32i
// CIR:     cir.store{{.*}} %[[RESULT]], %[[A_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:     cir.yield
// CIR:   }
// CIR: }

// CIR: cir.func private dso_local @__gxx_personality_v0(...) -> !s32i

// LLVM: define{{.*}} void @_Z30try_catch_with_empty_catch_allv()
// LLVM:   %[[A_ADDR:.*]] = alloca i32, align 4
// LLVM:   store i32 1, ptr %[[A_ADDR]], align 4
// LLVMCIR:   br label %[[BB_2:.*]]
// LLVMCIR: [[BB_2]]:
// LLVMCIR:   br label %[[BB_3:.*]]
// LLVMCIR: [[BB_3]]:
// LLVM:   ret void
// LLVMCIR: [[BB_4:.*]]:
// LLVMCIR:   %[[TMP_A:.*]] = load i32, ptr %[[A_ADDR]], align 4
// LLVMCIR:   %[[RESULT:.*]] = add nsw i32 %[[TMP_A]], 1
// LLVMCIR:   store i32 %[[RESULT]], ptr %[[A_ADDR]], align 4
// LLVMCIR:   br label %[[BB_7:.*]]
// LLVMCIR: [[BB_7]]:
// LLVMCIR:   br label %[[BB_8:.*]]
// LLVMCIR: [[BB_8]]:
// LLVMCIR:   ret void

void try_catch_with_empty_catch_all_2() {
  int a = 1;
  try {
    ++a;
    return;
  } catch (...) {
  }
}

// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} init : !cir.ptr<!s32i>
// CIR: %[[CONST_1:.*]] = cir.const #cir.int<1> : !s32i
// CIR: cir.store{{.*}} %[[CONST_1]], %[[A_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR: cir.scope {
// CIR:   cir.try {
// CIR:     %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:     %[[RESULT:.*]] = cir.inc nsw %[[TMP_A]] : !s32i
// CIR:     cir.store{{.*}} %[[RESULT]], %[[A_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:     cir.return
// CIR:   }
// CIR: }

// LLVM: define{{.*}} void @_Z32try_catch_with_empty_catch_all_2v()
// LLVM:   %[[A_ADDR:.*]] = alloca i32, align 4
// LLVM:   store i32 1, ptr %[[A_ADDR]], align 4
// LLVMCIR:   br label %[[BB_2:.*]]
// LLVMCIR: [[BB_2]]:
// LLVMCIR:   br label %[[BB_3:.*]]
// LLVMCIR: [[BB_3]]:
// LLVM:   %[[TMP_A:.*]] = load i32, ptr %[[A_ADDR]], align 4
// LLVM:   %[[RESULT:.*]] = add nsw i32 %[[TMP_A]], 1
// LLVM:   store i32 %[[RESULT]], ptr %[[A_ADDR]], align 4
// LLVM:   ret void
// LLVMCIR: [[BB_6:.*]]:
// LLVMCIR:   br label %[[BB_7:.*]]
// LLVMCIR: [[BB_7]]:
// LLVMCIR:   ret void

void try_catch_with_alloca() {
  try {
    int a;
    int b;
    int c = a + b;
  } catch (...) {
  }
}

// CIR: cir.func {{.*}} @_Z21try_catch_with_allocav() personality(@__gxx_personality_v0)
// CIR: cir.scope {
// CIR:   %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!s32i>
// CIR:   %[[B_ADDR:.*]] = cir.alloca "b" {{.*}} : !cir.ptr<!s32i>
// CIR:   %[[C_ADDR:.*]] = cir.alloca "c" {{.*}} init : !cir.ptr<!s32i>
// CIR:   cir.try {
// CIR:     %[[TMP_A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:     %[[TMP_B:.*]] = cir.load{{.*}} %[[B_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:     %[[RESULT:.*]] = cir.add nsw %[[TMP_A]], %[[TMP_B]] : !s32i
// CIR:     cir.store{{.*}} %[[RESULT]], %[[C_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:     cir.yield
// CIR:   }
// CIR: }

// LLVM: define{{.*}} void @_Z21try_catch_with_allocav()
// LLVM:   %[[A_ADDR:.*]] = alloca i32, align 4
// LLVM:   %[[B_ADDR:.*]] = alloca i32, align 4
// LLVM:   %[[C_ADDR:.*]] = alloca i32, align 4
// LLVMCIR:   br label %[[LABEL_1:.*]]
// LLVMCIR: [[LABEL_1]]:
// LLVMCIR:   br label %[[LABEL_2:.*]]
// LLVMCIR: [[LABEL_2]]:
// LLVM:   %[[TMP_A:.*]] = load i32, ptr %[[A_ADDR]], align 4
// LLVM:   %[[TMP_B:.*]] = load i32, ptr %[[B_ADDR]], align 4
// LLVM:   %[[RESULT:.*]] = add nsw i32 %[[TMP_A]], %[[TMP_B]]
// LLVM:   store i32 %[[RESULT]], ptr %[[C_ADDR]], align 4
// LLVMCIR:   br label %[[LABEL_3:.*]]
// LLVMCIR: [[LABEL_3]]:
// LLVMCIR:   br label %[[LABEL_4:.*]]
// LLVMCIR: [[LABEL_4]]:
// LLVM:   ret void

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

// LLVM: define{{.*}} void @_Z42calling_noexcept_function_inside_try_blockv()
// LLVMCIR:   br label %[[LABEL_1:.*]]
// LLVMCIR: [[LABEL_1]]:
// LLVMCIR:   br label %[[LABEL_2:.*]]
// LLVMCIR: [[LABEL_2]]:
// LLVM:   call void @_Z22function_with_noexceptv()
// LLVMCIR:   br label %[[LABEL_3:.*]]
// LLVMCIR: [[LABEL_3]]:
// LLVMCIR:   br label %[[LABEL_4:.*]]
// LLVMCIR: [[LABEL_4]]:
// LLVM:   ret void

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
// CIR:       } cleanup all {
// CIR:         cir.end_catch %[[CATCH_TOKEN]] : !cir.catch_token
// CIR:         cir.yield
// CIR:       }
// CIR:       cir.yield
// CIR:   }
// CIR: }

// LLVM: define {{.*}} void @_Z34call_function_inside_try_catch_allv() {{.*}} personality ptr @__gxx_personality_v0
// OGCG:   %[[EXN_SLOT:.*]] = alloca ptr, align 8
// OGCG:   %[[EH_SELECTOR_SLOT:.*]] = alloca i32, align 4
// LLVMCIR:   br label %[[TRY_SCOPE:.*]]
// LLVMCIR: [[TRY_SCOPE]]:
// LLVMCIR:   br label %[[TRY_BEGIN:.*]]
// LLVMCIR: [[TRY_BEGIN]]:
// LLVM:   %[[CALL:.*]] = invoke noundef i32 @_Z8divisionv()
// LLVM:           to label %[[INVOKE_CONT:.*]] unwind label %[[LANDING_PAD:.*]]
// LLVM: [[INVOKE_CONT]]:
// LLVM:   br label %[[TRY_CONT:.*]]
// LLVM: [[LANDING_PAD]]:
// LLVM:   %[[LP:.*]] = landingpad { ptr, i32 }
// LLVM:           catch ptr null
// LLVM:   %[[EXN_OBJ:.*]] = extractvalue { ptr, i32 } %[[LP]], 0
// OGCG:   store ptr %[[EXN_OBJ]], ptr %[[EXN_SLOT]], align 8
// LLVM:   %[[EH_SELECTOR_VAL:.*]] = extractvalue { ptr, i32 } %[[LP]], 1
// OGCG:   store i32 %[[EH_SELECTOR_VAL]], ptr %[[EH_SELECTOR_SLOT]], align 4
// LLVM:   br label %[[CATCH:.*]]
// LLVM: [[CATCH]]:
// LLVMCIR:   %[[EXN_OBJ_PHI:.*]] = phi ptr [ %[[EXN_OBJ]], %[[LANDING_PAD]] ]
// LLVMCIR:   %[[EH_SELECTOR_PHI:.*]] = phi i32 [ %[[EH_SELECTOR_VAL]], %[[LANDING_PAD]] ]
// LLVMCIR:   br label %[[BEGIN_CATCH:.*]]
// LLVMCIR: [[BEGIN_CATCH]]:
// LLVMCIR:   %[[EXN:.*]] = phi ptr [ %[[EXN_OBJ_PHI]], %[[CATCH]] ]
// LLVMCIR:   %{{.*}} = phi i32 [ %[[EH_SELECTOR_PHI]], %[[CATCH]] ]
// OGCG:   %[[EXN:.*]] = load ptr, ptr %[[EXN_SLOT]], align 8
// LLVM:   %[[EXN_PTR:.*]] = call ptr @__cxa_begin_catch(ptr %[[EXN]])
// LLVMCIR:   br label %[[CATCH_BODY:.*]]
// LLVMCIR: [[CATCH_BODY]]:
// LLVMCIR:   br label %[[END_CATCH:.*]]
// LLVMCIR: [[END_CATCH]]:
// LLVM:   call void @__cxa_end_catch()
// LLVMCIR:   br label %[[END_DISPATCH:.*]]
// LLVMCIR: [[END_DISPATCH]]:
// LLVMCIR:   br label %[[END_TRY:.*]]
// LLVMCIR: [[END_TRY]]:
// LLVM:   br label %[[TRY_CONT]]
// LLVM: [[TRY_CONT]]:
// LLVMCIR:   br label %[[DONE:.*]]
// LLVMCIR: [[DONE]]:
// LLVM:   ret void

void call_function_inside_try_catch_with_exception_type() {
  try {
    division();
  } catch (int e) {
  }
}

// CIR: cir.func {{.*}} @_Z50call_function_inside_try_catch_with_exception_typev() personality(@__gxx_personality_v0)
// CIR: cir.scope {
// CIR:   cir.try {
// CIR:     %[[CALL:.*]] = cir.call @_Z8divisionv()
// CIR:     cir.yield
// CIR:   } catch [type #cir.global_view<@_ZTIi> : !cir.ptr<!u8i>] (%[[EH_TOKEN:.*]]: !cir.eh_token{{.*}}) {
// CIR:     %[[CATCH_TOKEN:.*]], %[[EXN_PTR:.*]] = cir.begin_catch %[[EH_TOKEN]] : !cir.eh_token -> (!cir.catch_token, !cir.ptr<!void>)
// CIR:     cir.cleanup.scope {
// CIR:       cir.init_catch_param scalar %[[EXN_PTR]] to %{{.*}} : !cir.ptr<!void>, !cir.ptr<!s32i>
// CIR:       cir.yield
// CIR:     } cleanup all {
// CIR:       cir.end_catch %[[CATCH_TOKEN]] : !cir.catch_token
// CIR:       cir.yield
// CIR:     }
// CIR:     cir.yield
// CIR:   } unwind (%{{.*}}: !cir.eh_token{{.*}}) {
// CIR:     cir.resume %{{.*}} : !cir.eh_token
// CIR:   }
// CIR: }

// LLVM: define {{.*}} void @_Z50call_function_inside_try_catch_with_exception_typev() {{.*}} personality ptr @__gxx_personality_v0
// OGCG:   %[[EXN_SLOT:.*]] = alloca ptr, align 8
// OGCG:   %[[EH_SELECTOR_SLOT:.*]] = alloca i32, align 4
// LLVM:   %[[E_ADDR:.*]] = alloca i32, align 4
// LLVMCIR:   br label %[[TRY_SCOPE:.*]]
// LLVMCIR: [[TRY_SCOPE]]:
// LLVMCIR:   br label %[[TRY_BEGIN:.*]]
// LLVMCIR: [[TRY_BEGIN]]:
// LLVM:   %[[CALL:.*]] = invoke noundef i32 @_Z8divisionv()
// LLVM:           to label %[[INVOKE_CONT:.*]] unwind label %[[LANDING_PAD:.*]]
// LLVM: [[INVOKE_CONT]]:
// LLVM:   br label %[[TRY_CONT:.*]]
// LLVM: [[LANDING_PAD]]:
// LLVM:   %[[LP:.*]] = landingpad { ptr, i32 }
// LLVM:           catch ptr @_ZTIi
// LLVM:   %[[EXN_OBJ:.*]] = extractvalue { ptr, i32 } %[[LP]], 0
// OGCG:   store ptr %[[EXN_OBJ]], ptr %[[EXN_SLOT]], align 8
// LLVM:   %[[EH_SELECTOR_VAL:.*]] = extractvalue { ptr, i32 } %[[LP]], 1
// OGCG:   store i32 %[[EH_SELECTOR_VAL]], ptr %[[EH_SELECTOR_SLOT]], align 4
// LLVM:   br label %[[CATCH:.*]]
// LLVM: [[CATCH]]:
// LLVMCIR:   %[[EXN_OBJ_PHI:.*]] = phi ptr [ %[[EXN_OBJ]], %[[LANDING_PAD]] ]
// LLVMCIR:   %[[EH_SELECTOR_PHI:.*]] = phi i32 [ %[[EH_SELECTOR_VAL]], %[[LANDING_PAD]] ]
// LLVMCIR:   br label %[[DISPATCH:.*]]
// LLVMCIR: [[DISPATCH]]:
// LLVMCIR:   %[[DISPATCH_EXN:.*]] = phi ptr [ %[[EXN_OBJ_PHI]], %[[CATCH]] ]
// LLVMCIR:   %[[EH_SELECTOR:.*]] = phi i32 [ %[[EH_SELECTOR_PHI]], %[[CATCH]] ]
// OGCG:   %[[EH_SELECTOR:.*]] = load i32, ptr %[[EH_SELECTOR_SLOT]], align 4
// LLVM:   %[[EH_TYPE_ID:.*]] = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
// LLVM:   %[[TYPE_ID_EQ:.*]] = icmp eq i32 %[[EH_SELECTOR]], %[[EH_TYPE_ID]]
// LLVM:   br i1 %[[TYPE_ID_EQ]], label %[[BEGIN_CATCH:.*]], label %[[RESUME:.*]]
// LLVM: [[BEGIN_CATCH]]:
// LLVMCIR:   %[[EXN:.*]] = phi ptr [ %[[DISPATCH_EXN]], %[[DISPATCH]] ]
// LLVMCIR:   %{{.*}} = phi i32 [ %[[EH_SELECTOR]], %[[DISPATCH]] ]
// OGCG:   %[[EXN:.*]] = load ptr, ptr %[[EXN_SLOT]], align 8
// LLVM:   %[[EXN_PTR:.*]] = call ptr @__cxa_begin_catch(ptr %[[EXN]])
// LLVMCIR:   br label %[[CATCH_BODY:.*]]
// LLVMCIR: [[CATCH_BODY]]:
// LLVM:   %[[TMP_EXN:.*]] = load i32, ptr %[[EXN_PTR]], align 4
// LLVM:   store i32 %[[TMP_EXN]], ptr %[[E_ADDR]], align 4
// LLVMCIR:   br label %[[END_CATCH:.*]]
// LLVMCIR: [[END_CATCH]]:
// LLVM:   call void @__cxa_end_catch()
// LLVMCIR:   br label %[[END_DISPATCH:.*]]
// LLVMCIR: [[END_DISPATCH]]:
// LLVMCIR:   br label %[[END_TRY:.*]]
// LLVMCIR: [[END_TRY]]:
// LLVM:   br label %[[TRY_CONT]]
// OGCG: [[TRY_CONT]]:
// OGCG:   ret void
// LLVM: [[RESUME]]:
// LLVMCIR:   %[[RESUME_EXN:.*]] = phi ptr [ %[[DISPATCH_EXN]], %[[DISPATCH]] ]
// LLVMCIR:   %[[RESUME_SEL:.*]] = phi i32 [ %[[EH_SELECTOR]], %[[DISPATCH]] ]
// OGCG:   %[[RESUME_EXN:.*]] = load ptr, ptr %[[EXN_SLOT]], align 8
// OGCG:   %[[RESUME_SEL:.*]] = load i32, ptr %[[EH_SELECTOR_SLOT]], align 4
// LLVM:   %[[TMP_EXCEPTION_INFO:.*]] = insertvalue { ptr, i32 } poison, ptr %[[RESUME_EXN]], 0
// LLVM:   %[[EXCEPTION_INFO:.*]] = insertvalue { ptr, i32 } %[[TMP_EXCEPTION_INFO]], i32 %[[RESUME_SEL]], 1
// LLVM:   resume { ptr, i32 } %[[EXCEPTION_INFO]]
// LLVMCIR: [[TRY_CONT]]:
// LLVMCIR:   br label %[[DONE:.*]]
// LLVMCIR: [[DONE]]:
// LLVMCIR:   ret void

void call_function_inside_try_catch_with_ref_exception_type() {
  try {
    division();
  } catch (int &ref) {
  }
}

// CIR: cir.func {{.*}} @_Z54call_function_inside_try_catch_with_ref_exception_typev() personality(@__gxx_personality_v0)
// CIR: cir.scope {
// CIR:   cir.try {
// CIR:     %[[CALL:.*]] = cir.call @_Z8divisionv()
// CIR:     cir.yield
// CIR:   } catch [type #cir.global_view<@_ZTIi> : !cir.ptr<!u8i>] (%{{.*}}: !cir.eh_token {{.*}}) {
// CIR:     %[[CATCH_TOKEN:.*]], %[[EXN_PTR:.*]] = cir.begin_catch %{{.*}} : !cir.eh_token -> (!cir.catch_token, !cir.ptr<!void>)
// CIR:     cir.cleanup.scope {
// CIR:       cir.init_catch_param reference %[[EXN_PTR]] to %{{.*}} : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!s32i>>
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
// OGCG:   %[[EXN_SLOT:.*]] = alloca ptr, align 8
// OGCG:   %[[EH_SELECTOR_SLOT:.*]] = alloca i32, align 4
// LLVM:   %[[REF_ADDR:.*]] = alloca ptr, align 8
// LLVMCIR:   br label %[[TRY_SCOPE:.*]]
// LLVMCIR: [[TRY_SCOPE]]:
// LLVMCIR:   br label %[[TRY_BEGIN:.*]]
// LLVMCIR: [[TRY_BEGIN]]:
// LLVM:   %[[CALL:.*]] = invoke noundef i32 @_Z8divisionv()
// LLVM:           to label %[[INVOKE_CONT:.*]] unwind label %[[LANDING_PAD:.*]]
// LLVM: [[INVOKE_CONT]]:
// LLVM:   br label %[[TRY_CONT:.*]]
// LLVM: [[LANDING_PAD]]:
// LLVM:   %[[LP:.*]] = landingpad { ptr, i32 }
// LLVM:           catch ptr @_ZTIi
// LLVM:   %[[EXN_OBJ:.*]] = extractvalue { ptr, i32 } %[[LP]], 0
// OGCG:   store ptr %[[EXN_OBJ]], ptr %[[EXN_SLOT]], align 8
// LLVM:   %[[EH_SELECTOR_VAL:.*]] = extractvalue { ptr, i32 } %[[LP]], 1
// OGCG:   store i32 %[[EH_SELECTOR_VAL]], ptr %[[EH_SELECTOR_SLOT]], align 4
// LLVM:   br label %[[CATCH:.*]]
// LLVM: [[CATCH]]:
// LLVMCIR:   %[[EXN_OBJ_PHI:.*]] = phi ptr [ %[[EXN_OBJ]], %[[LANDING_PAD]] ]
// LLVMCIR:   %[[EH_SELECTOR_PHI:.*]] = phi i32 [ %[[EH_SELECTOR_VAL]], %[[LANDING_PAD]] ]
// LLVMCIR:   br label %[[DISPATCH:.*]]
// LLVMCIR: [[DISPATCH]]:
// LLVMCIR:   %[[DISPATCH_EXN:.*]] = phi ptr [ %[[EXN_OBJ_PHI]], %[[CATCH]] ]
// LLVMCIR:   %[[EH_SELECTOR:.*]] = phi i32 [ %[[EH_SELECTOR_PHI]], %[[CATCH]] ]
// OGCG:   %[[EH_SELECTOR:.*]] = load i32, ptr %[[EH_SELECTOR_SLOT]], align 4
// LLVM:   %[[EH_TYPE_ID:.*]] = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
// LLVM:   %[[TYPE_ID_EQ:.*]] = icmp eq i32 %[[EH_SELECTOR]], %[[EH_TYPE_ID]]
// LLVM:   br i1 %[[TYPE_ID_EQ]], label %[[BEGIN_CATCH:.*]], label %[[RESUME:.*]]
// LLVM: [[BEGIN_CATCH]]:
// LLVMCIR:   %[[EXN:.*]] = phi ptr [ %[[DISPATCH_EXN]], %[[DISPATCH]] ]
// LLVMCIR:   %{{.*}} = phi i32 [ %[[EH_SELECTOR]], %[[DISPATCH]] ]
// OGCG:   %[[EXN:.*]] = load ptr, ptr %[[EXN_SLOT]], align 8
// LLVM:   %[[EXN_PTR:.*]] = call ptr @__cxa_begin_catch(ptr %[[EXN]])
// LLVMCIR:   br label %[[CATCH_BODY:.*]]
// LLVMCIR: [[CATCH_BODY]]:
// LLVM:   store ptr %[[EXN_PTR]], ptr %[[REF_ADDR]], align 8
// LLVMCIR:   br label %[[END_CATCH:.*]]
// LLVMCIR: [[END_CATCH]]:
// LLVM:   call void @__cxa_end_catch()
// LLVMCIR:   br label %[[END_DISPATCH:.*]]
// LLVMCIR: [[END_DISPATCH]]:
// LLVMCIR:   br label %[[END_TRY:.*]]
// LLVMCIR: [[END_TRY]]:
// LLVM:   br label %[[TRY_CONT]]
// OGCG: [[TRY_CONT]]:
// OGCG:   ret void
// LLVM: [[RESUME]]:
// LLVMCIR:   %[[RESUME_EXN:.*]] = phi ptr [ %[[DISPATCH_EXN]], %[[DISPATCH]] ]
// LLVMCIR:   %[[RESUME_SEL:.*]] = phi i32 [ %[[EH_SELECTOR]], %[[DISPATCH]] ]
// OGCG:   %[[RESUME_EXN:.*]] = load ptr, ptr %[[EXN_SLOT]], align 8
// OGCG:   %[[RESUME_SEL:.*]] = load i32, ptr %[[EH_SELECTOR_SLOT]], align 4
// LLVM:   %[[TMP_EXCEPTION_INFO:.*]] = insertvalue { ptr, i32 } poison, ptr %[[RESUME_EXN]], 0
// LLVM:   %[[EXCEPTION_INFO:.*]] = insertvalue { ptr, i32 } %[[TMP_EXCEPTION_INFO]], i32 %[[RESUME_SEL]], 1
// LLVM:   resume { ptr, i32 } %[[EXCEPTION_INFO]]
// LLVMCIR: [[TRY_CONT]]:
// LLVMCIR:   br label %[[DONE:.*]]
// LLVMCIR: [[DONE]]:
// LLVMCIR:   ret void

void call_function_inside_try_catch_with_complex_exception_type() {
  try {
    division();
  } catch (int _Complex e) {
  }
}

// CIR: cir.func {{.*}} @_Z58call_function_inside_try_catch_with_complex_exception_typev() personality(@__gxx_personality_v0)
// CIR: cir.scope {
// CIR:   cir.try {
// CIR:     %[[CALL:.*]] = cir.call @_Z8divisionv()
// CIR:     cir.yield
// CIR:   } catch [type #cir.global_view<@_ZTICi> : !cir.ptr<!u8i>] (%[[EH_TOKEN:.*]]: !cir.eh_token{{.*}}) {
// CIR:     %[[CATCH_TOKEN:.*]], %[[EXN_PTR:.*]] = cir.begin_catch %[[EH_TOKEN]] : !cir.eh_token -> (!cir.catch_token, !cir.ptr<!void>)
// CIR:     cir.cleanup.scope {
// CIR:       cir.init_catch_param scalar %[[EXN_PTR]] to %{{.*}} : !cir.ptr<!void>, !cir.ptr<!cir.complex<!s32i>>
// CIR:       cir.yield
// CIR:     } cleanup all {
// CIR:       cir.end_catch %[[CATCH_TOKEN]] : !cir.catch_token
// CIR:       cir.yield
// CIR:     }
// CIR:     cir.yield
// CIR:   } unwind (%{{.*}}: !cir.eh_token{{.*}}) {
// CIR:     cir.resume %{{.*}} : !cir.eh_token
// CIR:   }
// CIR: }

// LLVM: define {{.*}} void @_Z58call_function_inside_try_catch_with_complex_exception_typev() {{.*}} personality ptr @__gxx_personality_v0
// OGCG:   %[[EXN_SLOT:.*]] = alloca ptr, align 8
// OGCG:   %[[EH_SELECTOR_SLOT:.*]] = alloca i32, align 4
// LLVM:   %[[E_ADDR:.*]] = alloca { i32, i32 }, align 4
// LLVMCIR:   br label %[[TRY_SCOPE:.*]]
// LLVMCIR: [[TRY_SCOPE]]:
// LLVMCIR:   br label %[[TRY_BEGIN:.*]]
// LLVMCIR: [[TRY_BEGIN]]:
// LLVM:   %[[CALL:.*]] = invoke noundef i32 @_Z8divisionv()
// LLVM:           to label %[[INVOKE_CONT:.*]] unwind label %[[LANDING_PAD:.*]]
// LLVM: [[INVOKE_CONT]]:
// LLVM:   br label %[[TRY_CONT:.*]]
// LLVM: [[LANDING_PAD]]:
// LLVM:   %[[LP:.*]] = landingpad { ptr, i32 }
// LLVM:           catch ptr @_ZTICi
// LLVM:   %[[EXN_OBJ:.*]] = extractvalue { ptr, i32 } %[[LP]], 0
// OGCG:   store ptr %[[EXN_OBJ]], ptr %[[EXN_SLOT]], align 8
// LLVM:   %[[EH_SELECTOR_VAL:.*]] = extractvalue { ptr, i32 } %[[LP]], 1
// OGCG:   store i32 %[[EH_SELECTOR_VAL]], ptr %[[EH_SELECTOR_SLOT]], align 4
// LLVM:   br label %[[CATCH:.*]]
// LLVM: [[CATCH]]:
// LLVMCIR:   %[[EXN_OBJ_PHI:.*]] = phi ptr [ %[[EXN_OBJ]], %[[LANDING_PAD]] ]
// LLVMCIR:   %[[EH_SELECTOR_PHI:.*]] = phi i32 [ %[[EH_SELECTOR_VAL]], %[[LANDING_PAD]] ]
// LLVMCIR:   br label %[[DISPATCH:.*]]
// LLVMCIR: [[DISPATCH]]:
// LLVMCIR:   %[[DISPATCH_EXN:.*]] = phi ptr [ %[[EXN_OBJ_PHI]], %[[CATCH]] ]
// LLVMCIR:   %[[EH_SELECTOR:.*]] = phi i32 [ %[[EH_SELECTOR_PHI]], %[[CATCH]] ]
// OGCG:   %[[EH_SELECTOR:.*]] = load i32, ptr %[[EH_SELECTOR_SLOT]], align 4
// LLVM:   %[[EH_TYPE_ID:.*]] = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTICi)
// LLVM:   %[[TYPE_ID_EQ:.*]] = icmp eq i32 %[[EH_SELECTOR]], %[[EH_TYPE_ID]]
// LLVM:   br i1 %[[TYPE_ID_EQ]], label %[[BEGIN_CATCH:.*]], label %[[RESUME:.*]]
// LLVM: [[BEGIN_CATCH]]:
// LLVMCIR:   %[[EXN:.*]] = phi ptr [ %[[DISPATCH_EXN]], %[[DISPATCH]] ]
// LLVMCIR:   %{{.*}} = phi i32 [ %[[EH_SELECTOR]], %[[DISPATCH]] ]
// OGCG:   %[[EXN:.*]] = load ptr, ptr %[[EXN_SLOT]], align 8
// LLVM:   %[[EXN_PTR:.*]] = call ptr @__cxa_begin_catch(ptr %[[EXN]])
// LLVMCIR:   br label %[[CATCH_BODY:.*]]
// LLVMCIR: [[CATCH_BODY]]:
// LLVMCIR:   %[[TMP_EXN:.*]] = load { i32, i32 }, ptr %[[EXN_PTR]], align 4
// LLVMCIR:   store { i32, i32 } %[[TMP_EXN]], ptr %[[E_ADDR]], align 4
// OGCG:   %[[EXCEPTION_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[EXN_PTR]], i32 0, i32 0
// OGCG:   %[[EXCEPTION_REAL:.*]] = load i32, ptr %[[EXCEPTION_REAL_PTR]], align 4
// OGCG:   %[[EXCEPTION_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[EXN_PTR]], i32 0, i32 1
// OGCG:   %[[EXCEPTION_IMAG:.*]] = load i32, ptr %[[EXCEPTION_IMAG_PTR]], align 4
// OGCG:   %[[E_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[E_ADDR]], i32 0, i32 0
// OGCG:   %[[E_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[E_ADDR]], i32 0, i32 1
// OGCG:   store i32 %[[EXCEPTION_REAL]], ptr %[[E_REAL_PTR]], align 4
// OGCG:   store i32 %[[EXCEPTION_IMAG]], ptr %[[E_IMAG_PTR]], align 4
// LLVMCIR:   br label %[[END_CATCH:.*]]
// LLVMCIR: [[END_CATCH]]:
// LLVM:   call void @__cxa_end_catch()
// LLVMCIR:   br label %[[END_DISPATCH:.*]]
// LLVMCIR: [[END_DISPATCH]]:
// LLVMCIR:   br label %[[END_TRY:.*]]
// LLVMCIR: [[END_TRY]]:
// LLVM:   br label %[[TRY_CONT]]
// OGCG: [[TRY_CONT]]:
// OGCG:   ret void
// LLVM: [[RESUME]]:
// LLVMCIR:   %[[RESUME_EXN:.*]] = phi ptr [ %[[DISPATCH_EXN]], %[[DISPATCH]] ]
// LLVMCIR:   %[[RESUME_SEL:.*]] = phi i32 [ %[[EH_SELECTOR]], %[[DISPATCH]] ]
// OGCG:   %[[RESUME_EXN:.*]] = load ptr, ptr %[[EXN_SLOT]], align 8
// OGCG:   %[[RESUME_SEL:.*]] = load i32, ptr %[[EH_SELECTOR_SLOT]], align 4
// LLVM:   %[[TMP_EXCEPTION_INFO:.*]] = insertvalue { ptr, i32 } poison, ptr %[[RESUME_EXN]], 0
// LLVM:   %[[EXCEPTION_INFO:.*]] = insertvalue { ptr, i32 } %[[TMP_EXCEPTION_INFO]], i32 %[[RESUME_SEL]], 1
// LLVM:   resume { ptr, i32 } %[[EXCEPTION_INFO]]
// LLVMCIR: [[TRY_CONT]]:
// LLVMCIR:   br label %[[DONE:.*]]
// LLVMCIR: [[DONE]]:
// LLVMCIR:   ret void

void call_function_inside_try_catch_with_array_exception_type() {
  try {
    division();
  } catch (int e[]) {
  }
}

// CIR: cir.func {{.*}} @_Z56call_function_inside_try_catch_with_array_exception_typev() personality(@__gxx_personality_v0)
// CIR: cir.scope {
// CIR:   cir.try {
// CIR:     %[[CALL:.*]] = cir.call @_Z8divisionv()
// CIR:     cir.yield
// CIR:   } catch [type #cir.global_view<@_ZTIPi> : !cir.ptr<!u8i>] (%[[EH_TOKEN:.*]]: !cir.eh_token{{.*}}) {
// CIR:     %[[CATCH_TOKEN:.*]], %[[EXN_PTR:.*]] = cir.begin_catch %[[EH_TOKEN]] : !cir.eh_token -> (!cir.catch_token, !cir.ptr<!void>)
// CIR:     cir.cleanup.scope {
// CIR:       cir.init_catch_param pointer %[[EXN_PTR]] to %{{.*}} : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!s32i>>
// CIR:       cir.yield
// CIR:     } cleanup all {
// CIR:       cir.end_catch %[[CATCH_TOKEN]] : !cir.catch_token
// CIR:       cir.yield
// CIR:     }
// CIR:     cir.yield
// CIR:   } unwind (%{{.*}}: !cir.eh_token{{.*}}) {
// CIR:     cir.resume %{{.*}} : !cir.eh_token
// CIR:   }
// CIR: }

// LLVM: define {{.*}} void @_Z56call_function_inside_try_catch_with_array_exception_typev() {{.*}} personality ptr @__gxx_personality_v0
// OGCG:   %[[EXN_SLOT:.*]] = alloca ptr, align 8
// OGCG:   %[[EH_SELECTOR_SLOT:.*]] = alloca i32, align 4
// LLVM:   %[[E_ADDR:.*]] = alloca ptr, align 8
// LLVMCIR:   br label %[[TRY_SCOPE:.*]]
// LLVMCIR: [[TRY_SCOPE]]:
// LLVMCIR:   br label %[[TRY_BEGIN:.*]]
// LLVMCIR: [[TRY_BEGIN]]:
// LLVM:   %[[CALL:.*]] = invoke noundef i32 @_Z8divisionv()
// LLVM:           to label %[[INVOKE_CONT:.*]] unwind label %[[LANDING_PAD:.*]]
// LLVM: [[INVOKE_CONT]]:
// LLVM:   br label %[[TRY_CONT:.*]]
// LLVM: [[LANDING_PAD]]:
// LLVM:   %[[LP:.*]] = landingpad { ptr, i32 }
// LLVM:           catch ptr @_ZTIPi
// LLVM:   %[[EXN_OBJ:.*]] = extractvalue { ptr, i32 } %[[LP]], 0
// OGCG:   store ptr %[[EXN_OBJ]], ptr %[[EXN_SLOT]], align 8
// LLVM:   %[[EH_SELECTOR_VAL:.*]] = extractvalue { ptr, i32 } %[[LP]], 1
// OGCG:   store i32 %[[EH_SELECTOR_VAL]], ptr %[[EH_SELECTOR_SLOT]], align 4
// LLVM:   br label %[[CATCH:.*]]
// LLVM: [[CATCH]]:
// LLVMCIR:   %[[EXN_OBJ_PHI:.*]] = phi ptr [ %[[EXN_OBJ]], %[[LANDING_PAD]] ]
// LLVMCIR:   %[[EH_SELECTOR_PHI:.*]] = phi i32 [ %[[EH_SELECTOR_VAL]], %[[LANDING_PAD]] ]
// LLVMCIR:   br label %[[DISPATCH:.*]]
// LLVMCIR: [[DISPATCH]]:
// LLVMCIR:   %[[DISPATCH_EXN:.*]] = phi ptr [ %[[EXN_OBJ_PHI]], %[[CATCH]] ]
// LLVMCIR:   %[[EH_SELECTOR:.*]] = phi i32 [ %[[EH_SELECTOR_PHI]], %[[CATCH]] ]
// OGCG:   %[[EH_SELECTOR:.*]] = load i32, ptr %[[EH_SELECTOR_SLOT]], align 4
// LLVM:   %[[EH_TYPE_ID:.*]] = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIPi)
// LLVM:   %[[TYPE_ID_EQ:.*]] = icmp eq i32 %[[EH_SELECTOR]], %[[EH_TYPE_ID]]
// LLVM:   br i1 %[[TYPE_ID_EQ]], label %[[BEGIN_CATCH:.*]], label %[[RESUME:.*]]
// LLVM: [[BEGIN_CATCH]]:
// LLVMCIR:   %[[EXN:.*]] = phi ptr [ %[[DISPATCH_EXN]], %[[DISPATCH]] ]
// LLVMCIR:   %{{.*}} = phi i32 [ %[[EH_SELECTOR]], %[[DISPATCH]] ]
// OGCG:   %[[EXN:.*]] = load ptr, ptr %[[EXN_SLOT]], align 8
// LLVM:   %[[EXN_PTR:.*]] = call ptr @__cxa_begin_catch(ptr %[[EXN]])
// LLVMCIR:   br label %[[CATCH_BODY:.*]]
// LLVMCIR: [[CATCH_BODY]]:
// LLVM:   store ptr %[[EXN_PTR]], ptr %[[E_ADDR]], align 8
// LLVMCIR:   br label %[[END_CATCH:.*]]
// LLVMCIR: [[END_CATCH]]:
// LLVM:   call void @__cxa_end_catch()
// LLVMCIR:   br label %[[END_DISPATCH:.*]]
// LLVMCIR: [[END_DISPATCH]]:
// LLVMCIR:   br label %[[END_TRY:.*]]
// LLVMCIR: [[END_TRY]]:
// LLVM:   br label %[[TRY_CONT]]
// OGCG: [[TRY_CONT]]:
// OGCG:   ret void
// LLVM: [[RESUME]]:
// LLVMCIR:   %[[RESUME_EXN:.*]] = phi ptr [ %[[DISPATCH_EXN]], %[[DISPATCH]] ]
// LLVMCIR:   %[[RESUME_SEL:.*]] = phi i32 [ %[[EH_SELECTOR]], %[[DISPATCH]] ]
// OGCG:   %[[RESUME_EXN:.*]] = load ptr, ptr %[[EXN_SLOT]], align 8
// OGCG:   %[[RESUME_SEL:.*]] = load i32, ptr %[[EH_SELECTOR_SLOT]], align 4
// LLVM:   %[[TMP_EXCEPTION_INFO:.*]] = insertvalue { ptr, i32 } poison, ptr %[[RESUME_EXN]], 0
// LLVM:   %[[EXCEPTION_INFO:.*]] = insertvalue { ptr, i32 } %[[TMP_EXCEPTION_INFO]], i32 %[[RESUME_SEL]], 1
// LLVM:   resume { ptr, i32 } %[[EXCEPTION_INFO]]
// LLVMCIR: [[TRY_CONT]]:
// LLVMCIR:   br label %[[DONE:.*]]
// LLVMCIR: [[DONE]]:
// LLVMCIR:   ret void

void call_function_inside_try_catch_with_exception_type_and_catch_all() {
  try {
    division();
  } catch (int e) {
  } catch (...) {
  }
}

// CIR: cir.func {{.*}} @_Z64call_function_inside_try_catch_with_exception_type_and_catch_allv() personality(@__gxx_personality_v0)
// CIR: cir.scope {
// CIR:   cir.try {
// CIR:     %[[CALL:.*]] = cir.call @_Z8divisionv()
// CIR:     cir.yield
// CIR:   } catch [type #cir.global_view<@_ZTIi> : !cir.ptr<!u8i>] (%[[EH_TOKEN:.*]]: !cir.eh_token{{.*}}) {
// CIR:     %[[CATCH_TOKEN:.*]], %[[EXN_PTR:.*]] = cir.begin_catch %[[EH_TOKEN]] : !cir.eh_token -> (!cir.catch_token, !cir.ptr<!void>)
// CIR:     cir.cleanup.scope {
// CIR:       cir.init_catch_param scalar %[[EXN_PTR]] to %{{.*}} : !cir.ptr<!void>, !cir.ptr<!s32i>
// CIR:       cir.yield
// CIR:     } cleanup all {
// CIR:       cir.end_catch %[[CATCH_TOKEN]] : !cir.catch_token
// CIR:       cir.yield
// CIR:     }
// CIR:     cir.yield
// CIR:   } catch all (%[[EH_TOKEN2:.*]]: !cir.eh_token{{.*}}) {
// CIR:     %[[CATCH_TOKEN2:.*]], %{{.*}} = cir.begin_catch %[[EH_TOKEN2]] : !cir.eh_token -> (!cir.catch_token, !cir.ptr<!void>)
// CIR:     cir.cleanup.scope {
// CIR:       cir.yield
// CIR:     } cleanup all {
// CIR:       cir.end_catch %[[CATCH_TOKEN2]] : !cir.catch_token
// CIR:       cir.yield
// CIR:     }
// CIR:     cir.yield
// CIR:   }
// CIR: }

// LLVM: define {{.*}} void @_Z64call_function_inside_try_catch_with_exception_type_and_catch_allv() {{.*}} personality ptr @__gxx_personality_v0
// OGCG:   %[[EXN_SLOT:.*]] = alloca ptr, align 8
// OGCG:   %[[EH_SELECTOR_SLOT:.*]] = alloca i32, align 4
// LLVM:   %[[E_ADDR:.*]] = alloca i32, align 4
// LLVMCIR:   br label %[[TRY_SCOPE:.*]]
// LLVMCIR: [[TRY_SCOPE]]:
// LLVMCIR:   br label %[[TRY_BEGIN:.*]]
// LLVMCIR: [[TRY_BEGIN]]:
// LLVM:   %[[CALL:.*]] = invoke noundef i32 @_Z8divisionv()
// LLVM:           to label %[[INVOKE_CONT:.*]] unwind label %[[LANDING_PAD:.*]]
// LLVM: [[INVOKE_CONT]]:
// LLVM:   br label %[[TRY_CONT:.*]]
// LLVM: [[LANDING_PAD]]:
// LLVM:   %[[LP:.*]] = landingpad { ptr, i32 }
// LLVM:           catch ptr @_ZTIi
// LLVM:           catch ptr null
// LLVM:   %[[EXN_OBJ:.*]] = extractvalue { ptr, i32 } %[[LP]], 0
// OGCG:   store ptr %[[EXN_OBJ]], ptr %[[EXN_SLOT]], align 8
// LLVM:   %[[EH_SELECTOR_VAL:.*]] = extractvalue { ptr, i32 } %[[LP]], 1
// OGCG:   store i32 %[[EH_SELECTOR_VAL]], ptr %[[EH_SELECTOR_SLOT]], align 4
// LLVM:   br label %[[CATCH:.*]]
// LLVM: [[CATCH]]:
// LLVMCIR:   %[[EXN_OBJ_PHI:.*]] = phi ptr [ %[[EXN_OBJ]], %[[LANDING_PAD]] ]
// LLVMCIR:   %[[EH_SELECTOR_PHI:.*]] = phi i32 [ %[[EH_SELECTOR_VAL]], %[[LANDING_PAD]] ]
// LLVMCIR:   br label %[[DISPATCH:.*]]
// LLVMCIR: [[DISPATCH]]:
// LLVMCIR:   %[[DISPATCH_EXN:.*]] = phi ptr [ %[[EXN_OBJ_PHI]], %[[CATCH]] ]
// LLVMCIR:   %[[EH_SELECTOR:.*]] = phi i32 [ %[[EH_SELECTOR_PHI]], %[[CATCH]] ]
// OGCG:   %[[EH_SELECTOR:.*]] = load i32, ptr %[[EH_SELECTOR_SLOT]], align 4
// LLVM:   %[[EH_TYPE_ID:.*]] = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
// LLVM:   %[[TYPE_ID_EQ:.*]] = icmp eq i32 %[[EH_SELECTOR]], %[[EH_TYPE_ID]]
// LLVM:   br i1 %[[TYPE_ID_EQ]], label %[[BEGIN_CATCH:.*]], label %[[CATCH_ALL:.*]]
// LLVM: [[BEGIN_CATCH]]:
// LLVMCIR:   %[[EXN:.*]] = phi ptr [ %[[DISPATCH_EXN]], %[[DISPATCH]] ]
// LLVMCIR:   %{{.*}} = phi i32 [ %[[EH_SELECTOR]], %[[DISPATCH]] ]
// OGCG:   %[[EXN:.*]] = load ptr, ptr %[[EXN_SLOT]], align 8
// LLVM:   %[[EXN_PTR:.*]] = call ptr @__cxa_begin_catch(ptr %[[EXN]])
// LLVMCIR:   br label %[[CATCH_BODY:.*]]
// LLVMCIR: [[CATCH_BODY]]:
// LLVM:   %[[TMP_EXN:.*]] = load i32, ptr %[[EXN_PTR]], align 4
// LLVM:   store i32 %[[TMP_EXN]], ptr %[[E_ADDR]], align 4
// LLVMCIR:   br label %[[END_CATCH:.*]]
// LLVMCIR: [[END_CATCH]]:
// LLVM:   call void @__cxa_end_catch()
// LLVMCIR:   br label %[[END_DISPATCH:.*]]
// LLVMCIR: [[END_DISPATCH]]:
// LLVMCIR:   br label %[[END_TRY:.*]]
// LLVMCIR: [[END_TRY]]:
// LLVM:   br label %[[TRY_CONT]]
// OGCG: [[TRY_CONT]]:
// OGCG:   ret void
// LLVM: [[CATCH_ALL]]:
// LLVMCIR:   %[[CATCH_ALL_EXN:.*]] = phi ptr [ %[[DISPATCH_EXN]], %[[DISPATCH]] ]
// LLVMCIR:   %{{.*}} = phi i32 [ %[[EH_SELECTOR]], %[[DISPATCH]] ]
// OGCG:   %[[CATCH_ALL_EXN:.*]] = load ptr, ptr %[[EXN_SLOT]], align 8
// LLVM:   %[[CATCH_ALL_EXN_PTR:.*]] = call ptr @__cxa_begin_catch(ptr %[[CATCH_ALL_EXN]])
// LLVMCIR:   br label %[[CATCH_ALL_BODY:.*]]
// LLVMCIR: [[CATCH_ALL_BODY]]:
// LLVMCIR:   br label %[[END_CATCH_ALL:.*]]
// LLVMCIR: [[END_CATCH_ALL]]:
// LLVM:   call void @__cxa_end_catch()
// LLVMCIR:   br label %[[END_DISPATCH_ALL:.*]]
// LLVMCIR: [[END_DISPATCH_ALL]]:
// LLVMCIR:   br label %[[END_TRY_ALL:.*]]
// LLVMCIR: [[END_TRY_ALL]]:
// LLVM:   br label %[[TRY_CONT]]
// LLVMCIR: [[TRY_CONT]]:
// LLVMCIR:   br label %[[DONE:.*]]
// LLVMCIR: [[DONE]]:
// LLVMCIR:   ret void

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

// CIR: cir.func {{.*}} @_Z23cleanup_inside_try_bodyv(){{.*}} personality(@__gxx_personality_v0){{.*}} {
// CIR: cir.scope {
// CIR:   %[[S:.*]] = cir.alloca "s" {{.*}} : !cir.ptr<!rec_S>
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
// LLVM:   %[[S:.*]] = alloca %struct.S, align 1
// OGCG:   %[[EXN_SLOT:.*]] = alloca ptr, align 8
// OGCG:   %[[EH_SELECTOR_SLOT:.*]] = alloca i32, align 4
// LLVMCIR:   br label %[[TRY_SCOPE:.*]]
// LLVMCIR: [[TRY_SCOPE]]:
// LLVMCIR:   br label %[[TRY_BEGIN:.*]]
// LLVMCIR: [[TRY_BEGIN]]:
// LLVMCIR:   br label %[[CLEANUP_SCOPE:.*]]
// LLVMCIR: [[CLEANUP_SCOPE]]:
// LLVM:   %[[CALL:.*]] = invoke noundef i32 @_Z8divisionv()
// LLVM:           to label %[[INVOKE_CONT:.*]] unwind label %[[LANDING_PAD:.*]]
// LLVM: [[INVOKE_CONT]]:
// LLVMCIR:   br label %[[CLEANUP:.*]]
// LLVMCIR: [[CLEANUP]]:
// LLVM:   call void @_ZN1SD1Ev(ptr noundef nonnull align 1 dereferenceable(1) %[[S]])
// LLVMCIR:   br label %[[END_CLEANUP:.*]]
// LLVMCIR: [[END_CLEANUP]]:
// LLVM:   br label %[[TRY_CONT:.*]]
// LLVM: [[LANDING_PAD]]:
// LLVM:   %[[LP:.*]] = landingpad { ptr, i32 }
// LLVM:           catch ptr null
// LLVM:   %[[EXN_OBJ:.*]] = extractvalue { ptr, i32 } %[[LP]], 0
// OGCG:   store ptr %[[EXN_OBJ]], ptr %[[EXN_SLOT]], align 8
// LLVM:   %[[EH_SELECTOR_VAL:.*]] = extractvalue { ptr, i32 } %[[LP]], 1
// OGCG:   store i32 %[[EH_SELECTOR_VAL]], ptr %[[EH_SELECTOR_SLOT]], align 4
// LLVMCIR:   br label %[[CLEANUP_LANDING:.*]]
// LLVMCIR: [[CLEANUP_LANDING]]:
// LLVMCIR:   %[[EXN_OBJ_PHI:.*]] = phi ptr [ %[[EXN_OBJ]], %[[LANDING_PAD]] ]
// LLVMCIR:   %[[EH_SELECTOR_PHI:.*]] = phi i32 [ %[[EH_SELECTOR_VAL]], %[[LANDING_PAD]] ]
// LLVM:   call void @_ZN1SD1Ev(ptr noundef nonnull align 1 dereferenceable(1) %[[S]])
// LLVM:   br label %[[CATCH:.*]]
// LLVMCIR: [[TRY_CONT]]:
// LLVMCIR-NEXT:   br label %[[CATCH_CONT:.*]]
// LLVM: [[CATCH]]:
// LLVMCIR:   %[[CATCH_EXN:.*]] = phi ptr [ %[[EXN_OBJ_PHI]], %[[CLEANUP_LANDING]] ]
// LLVMCIR:   %[[CATCH_SEL:.*]] = phi i32 [ %[[EH_SELECTOR_PHI]], %[[CLEANUP_LANDING]] ]
// LLVMCIR:   br label %[[BEGIN_CATCH:.*]]
// LLVMCIR: [[BEGIN_CATCH]]:
// LLVMCIR:   %[[EXN:.*]] = phi ptr [ %[[CATCH_EXN]], %[[CATCH]] ]
// LLVMCIR:   %{{.*}} = phi i32 [ %[[CATCH_SEL]], %[[CATCH]] ]
// OGCG:   %[[EXN:.*]] = load ptr, ptr %[[EXN_SLOT]], align 8
// LLVM:   %[[EXN_PTR:.*]] = call ptr @__cxa_begin_catch(ptr %[[EXN]])
// LLVMCIR:   br label %[[CATCH_BODY:.*]]
// LLVMCIR: [[CATCH_BODY]]:
// LLVMCIR:   br label %[[END_CATCH:.*]]
// LLVMCIR: [[END_CATCH]]:
// LLVM:   call void @__cxa_end_catch()
// LLVMCIR:   br label %[[END_DISPATCH:.*]]
// LLVMCIR: [[END_DISPATCH]]:
// LLVMCIR:   br label %[[END_TRY:.*]]
// LLVMCIR: [[END_TRY]]:
// LLVMCIR:   br label %[[CATCH_CONT]]
// LLVMCIR: [[CATCH_CONT]]:
// LLVMCIR:   br label %[[DONE:.*]]
// LLVMCIR: [[DONE]]:
// OGCG:   br label %[[TRY_CONT]]
// OGCG: [[TRY_CONT]]:
// LLVM:   ret void

struct CustomError {
  int error_code;
};

void call_function_inside_try_catch_with_aggregate_exception_type() {
  try {
    division();
  } catch (CustomError e) {
  }
}


// CIR: cir.func {{.*}} @_Z60call_function_inside_try_catch_with_aggregate_exception_typev(){{.*}} personality(@__gxx_personality_v0){{.*}} {
// CIR: cir.scope {
// CIR:   cir.try {
// CIR:     %[[CALL:.*]] = cir.call @_Z8divisionv() : () -> (!s32i {llvm.noundef})
// CIR:     cir.yield
// CIR:   } catch [type #cir.global_view<@_ZTI11CustomError> : !cir.ptr<!u8i>] (%{{.*}}: !cir.eh_token {{.*}}) {
// CIR:     %[[CATCH_TOKEN:.*]], %[[EXN_PTR:.*]] = cir.begin_catch %{{.*}} : !cir.eh_token -> (!cir.catch_token, !cir.ptr<!void>)
// CIR:     cir.cleanup.scope {
// CIR:       cir.init_catch_param trivial_copy %[[EXN_PTR]] to %{{.*}} : !cir.ptr<!void>, !cir.ptr<!rec_CustomError>
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

// LLVM: define {{.*}} void @_Z60call_function_inside_try_catch_with_aggregate_exception_typev() {{.*}} personality ptr @__gxx_personality_v0
// OGCG:   %[[EXN_SLOT:.*]] = alloca ptr, align 8
// OGCG:   %[[EH_SELECTOR_SLOT:.*]] = alloca i32, align 4
// LLVM:   %[[E_ADDR:.*]] = alloca %struct.CustomError, align 4
// LLVMCIR:   br label %[[TRY_SCOPE:.*]]
// LLVMCIR: [[TRY_SCOPE]]:
// LLVMCIR:   br label %[[TRY_BEGIN:.*]]
// LLVMCIR: [[TRY_BEGIN]]:
// LLVM:   %[[CALL:.*]] = invoke noundef i32 @_Z8divisionv()
// LLVM:           to label %[[INVOKE_CONT:.*]] unwind label %[[LANDING_PAD:.*]]
// LLVM: [[INVOKE_CONT]]:
// LLVM:   br label %[[TRY_CONT:.*]]
// LLVM: [[LANDING_PAD]]:
// LLVM:   %[[LP:.*]] = landingpad { ptr, i32 }
// LLVM:           catch ptr @_ZTI11CustomError
// LLVM:   %[[EXN_OBJ:.*]] = extractvalue { ptr, i32 } %[[LP]], 0
// OGCG:   store ptr %[[EXN_OBJ]], ptr %[[EXN_SLOT]], align 8
// LLVM:   %[[EH_SELECTOR_VAL:.*]] = extractvalue { ptr, i32 } %[[LP]], 1
// OGCG:   store i32 %[[EH_SELECTOR_VAL]], ptr %[[EH_SELECTOR_SLOT]], align 4
// LLVM:   br label %[[CATCH:.*]]
// LLVM: [[CATCH]]:
// LLVMCIR:   %[[EXN_OBJ_PHI:.*]] = phi ptr [ %[[EXN_OBJ]], %[[LANDING_PAD]] ]
// LLVMCIR:   %[[EH_SELECTOR_PHI:.*]] = phi i32 [ %[[EH_SELECTOR_VAL]], %[[LANDING_PAD]] ]
// LLVMCIR:   br label %[[DISPATCH:.*]]
// LLVMCIR: [[DISPATCH]]:
// LLVMCIR:   %[[DISPATCH_EXN:.*]] = phi ptr [ %[[EXN_OBJ_PHI]], %[[CATCH]] ]
// LLVMCIR:   %[[EH_SELECTOR:.*]] = phi i32 [ %[[EH_SELECTOR_PHI]], %[[CATCH]] ]
// OGCG:   %[[EH_SELECTOR:.*]] = load i32, ptr %[[EH_SELECTOR_SLOT]], align 4
// LLVM:   %[[EH_TYPE_ID:.*]] = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTI11CustomError)
// LLVM:   %[[TYPE_ID_EQ:.*]] = icmp eq i32 %[[EH_SELECTOR]], %[[EH_TYPE_ID]]
// LLVM:   br i1 %[[TYPE_ID_EQ]], label %[[BEGIN_CATCH:.*]], label %[[RESUME:.*]]
// LLVM: [[BEGIN_CATCH]]:
// LLVMCIR:   %[[EXN:.*]] = phi ptr [ %[[DISPATCH_EXN]], %[[DISPATCH]] ]
// LLVMCIR:   %{{.*}} = phi i32 [ %[[EH_SELECTOR]], %[[DISPATCH]] ]
// OGCG:   %[[EXN:.*]] = load ptr, ptr %[[EXN_SLOT]], align 8
// LLVM:   %[[EXN_PTR:.*]] = call ptr @__cxa_begin_catch(ptr %[[EXN]])
// LLVMCIR:   br label %[[CATCH_BODY:.*]]
// LLVMCIR: [[CATCH_BODY]]:
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[E_ADDR]], ptr align 4 %[[EXN_PTR]], i64 4, i1 false)
// LLVMCIR:   br label %[[END_CATCH:.*]]
// LLVMCIR: [[END_CATCH]]:
// LLVM:   call void @__cxa_end_catch()
// LLVMCIR:   br label %[[END_DISPATCH:.*]]
// LLVMCIR: [[END_DISPATCH]]:
// LLVMCIR:   br label %[[END_TRY:.*]]
// LLVMCIR: [[END_TRY]]:
// LLVM:   br label %[[TRY_CONT]]
// OGCG: [[TRY_CONT]]:
// OGCG:   ret void
// LLVM: [[RESUME]]:
// LLVMCIR:   %[[RESUME_EXN:.*]] = phi ptr [ %[[DISPATCH_EXN]], %[[DISPATCH]] ]
// LLVMCIR:   %[[RESUME_SEL:.*]] = phi i32 [ %[[EH_SELECTOR]], %[[DISPATCH]] ]
// OGCG:   %[[RESUME_EXN:.*]] = load ptr, ptr %[[EXN_SLOT]], align 8
// OGCG:   %[[RESUME_SEL:.*]] = load i32, ptr %[[EH_SELECTOR_SLOT]], align 4
// LLVM:   %[[TMP_EXCEPTION_INFO:.*]] = insertvalue { ptr, i32 } poison, ptr %[[RESUME_EXN]], 0
// LLVM:   %[[EXCEPTION_INFO:.*]] = insertvalue { ptr, i32 } %[[TMP_EXCEPTION_INFO]], i32 %[[RESUME_SEL]], 1
// LLVM:   resume { ptr, i32 } %[[EXCEPTION_INFO]]
// LLVMCIR: [[TRY_CONT]]:
// LLVMCIR:   br label %[[DONE:.*]]
// LLVMCIR: [[DONE]]:
// LLVMCIR:   ret void

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

// CIR: cir.func {{.*}} @_Z68call_function_inside_try_catch_with_ref_ptr_of_record_exception_typev(){{.*}} personality(@__gxx_personality_v0){{.*}} {
// CIR:   %[[E_ADDR:.*]] = cir.alloca "ref_ptr" {{.*}} const : !cir.ptr<!cir.ptr<!cir.ptr<!rec_Record>>>
// CIR:   cir.try {
// CIR:     %[[CALL:.*]] = cir.call @_Z8divisionv() : () -> (!s32i {llvm.noundef})
// CIR:     cir.yield
// CIR:   } catch [type #cir.global_view<@_ZTIP6Record> : !cir.ptr<!u8i>] (%[[EH_TOKEN:.*]]: !cir.eh_token {{.*}}) {
// CIR-NEXT:     %[[CATCH_TOKEN:.*]], %[[EXN_PTR:.*]] = cir.begin_catch %[[EH_TOKEN]] : !cir.eh_token -> (!cir.catch_token, !cir.ptr<!void>)
// CIR:     cir.cleanup.scope {
// CIR:       cir.init_catch_param reference_to_record_pointer %[[EXN_PTR]] to %[[E_ADDR]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!cir.ptr<!rec_Record>>>
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
// LLVMCIR:   %[[EXN_BYREF_TMP:.*]] = alloca ptr, align 8
// LLVMCIR:   %[[E_ADDR:.*]] = alloca ptr, align 8
// OGCG:   %[[EXN_SLOT:.*]] = alloca ptr, align 8
// OGCG:   %[[EH_SELECTOR_SLOT:.*]] = alloca i32, align 4
// OGCG:   %[[E_ADDR:.*]] = alloca ptr, align 8
// OGCG:   %[[EXN_BYREF_TMP:.*]] = alloca ptr, align 8
// LLVMCIR:   br label %[[TRY_SCOPE:.*]]
// LLVMCIR: [[TRY_SCOPE]]:
// LLVMCIR:   br label %[[TRY_BEGIN:.*]]
// LLVMCIR: [[TRY_BEGIN]]:
// LLVM:   %[[CALL:.*]] = invoke noundef i32 @_Z8divisionv()
// LLVM:           to label %[[INVOKE_CONT:.*]] unwind label %[[LANDING_PAD:.*]]
// LLVM: [[INVOKE_CONT]]:
// LLVM:   br label %[[TRY_CONT:.*]]
// LLVM: [[LANDING_PAD]]:
// LLVM:   %[[LP:.*]] = landingpad { ptr, i32 }
// LLVM:           catch ptr @_ZTIP6Record
// LLVM:   %[[EXN_OBJ:.*]] = extractvalue { ptr, i32 } %[[LP]], 0
// OGCG:   store ptr %[[EXN_OBJ]], ptr %[[EXN_SLOT]], align 8
// LLVM:   %[[EH_SELECTOR_VAL:.*]] = extractvalue { ptr, i32 } %[[LP]], 1
// OGCG:   store i32 %[[EH_SELECTOR_VAL]], ptr %[[EH_SELECTOR_SLOT]], align 4
// LLVM:   br label %[[CATCH:.*]]
// LLVM: [[CATCH]]:
// LLVMCIR:   %[[EXN_OBJ_PHI:.*]] = phi ptr [ %[[EXN_OBJ]], %[[LANDING_PAD]] ]
// LLVMCIR:   %[[EH_SELECTOR_PHI:.*]] = phi i32 [ %[[EH_SELECTOR_VAL]], %[[LANDING_PAD]] ]
// LLVMCIR:   br label %[[DISPATCH:.*]]
// LLVMCIR: [[DISPATCH]]:
// LLVMCIR:   %[[DISPATCH_EXN:.*]] = phi ptr [ %[[EXN_OBJ_PHI]], %[[CATCH]] ]
// LLVMCIR:   %[[EH_SELECTOR:.*]] = phi i32 [ %[[EH_SELECTOR_PHI]], %[[CATCH]] ]
// OGCG:   %[[EH_SELECTOR:.*]] = load i32, ptr %[[EH_SELECTOR_SLOT]], align 4
// LLVM:   %[[EH_TYPE_ID:.*]] = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIP6Record)
// LLVM:   %[[TYPE_ID_EQ:.*]] = icmp eq i32 %[[EH_SELECTOR]], %[[EH_TYPE_ID]]
// LLVM:   br i1 %[[TYPE_ID_EQ]], label %[[BEGIN_CATCH:.*]], label %[[RESUME:.*]]
// LLVM: [[BEGIN_CATCH]]:
// LLVMCIR:   %[[EXN:.*]] = phi ptr [ %[[DISPATCH_EXN]], %[[DISPATCH]] ]
// LLVMCIR:   %{{.*}} = phi i32 [ %[[EH_SELECTOR]], %[[DISPATCH]] ]
// OGCG:   %[[EXN:.*]] = load ptr, ptr %[[EXN_SLOT]], align 8
// LLVM-NEXT:   %[[EXN_PTR:.*]] = call ptr @__cxa_begin_catch(ptr %[[EXN]])
// LLVMCIR:   br label %[[CATCH_BODY:.*]]
// LLVMCIR: [[CATCH_BODY]]:
// LLVM-NEXT:   store ptr %[[EXN_PTR]], ptr %[[EXN_BYREF_TMP]], align 8
// LLVM-NEXT:   store ptr %[[EXN_BYREF_TMP]], ptr %[[E_ADDR]], align 8
// LLVMCIR-NEXT:   br label %[[END_CATCH:.*]]
// LLVMCIR: [[END_CATCH]]:
// LLVM:   call void @__cxa_end_catch()
// LLVMCIR:   br label %[[END_DISPATCH:.*]]
// LLVMCIR: [[END_DISPATCH]]:
// LLVMCIR:   br label %[[END_TRY:.*]]
// LLVMCIR: [[END_TRY]]:
// LLVM:   br label %[[TRY_CONT]]
// OGCG: [[TRY_CONT]]:
// OGCG:   ret void
// LLVM: [[RESUME]]:
// LLVMCIR:   %[[RESUME_EXN:.*]] = phi ptr [ %[[DISPATCH_EXN]], %[[DISPATCH]] ]
// LLVMCIR:   %[[RESUME_SEL:.*]] = phi i32 [ %[[EH_SELECTOR]], %[[DISPATCH]] ]
// OGCG:   %[[RESUME_EXN:.*]] = load ptr, ptr %[[EXN_SLOT]], align 8
// OGCG:   %[[RESUME_SEL:.*]] = load i32, ptr %[[EH_SELECTOR_SLOT]], align 4
// LLVM:   %[[TMP_EXCEPTION_INFO:.*]] = insertvalue { ptr, i32 } poison, ptr %[[RESUME_EXN]], 0
// LLVM:   %[[EXCEPTION_INFO:.*]] = insertvalue { ptr, i32 } %[[TMP_EXCEPTION_INFO]], i32 %[[RESUME_SEL]], 1
// LLVM:   resume { ptr, i32 } %[[EXCEPTION_INFO]]
// LLVMCIR: [[TRY_CONT]]:
// LLVMCIR:   br label %[[DONE:.*]]
// LLVMCIR: [[DONE]]:
// LLVMCIR:   ret void

void call_function_inside_try_catch_with_exception_member_ptr_type() {
  try {
    division();
  } catch (int Record::*memberPtr) {
  }
}

// CIR: cir.func {{.*}} @_Z61call_function_inside_try_catch_with_exception_member_ptr_typev(){{.*}} personality(@__gxx_personality_v0){{.*}} {
// CIR: cir.scope {
// CIR:   cir.try {
// CIR:     %[[CALL:.*]] = cir.call @_Z8divisionv() : () -> (!s32i {llvm.noundef})
// CIR:     cir.yield
// CIR:   } catch [type #cir.global_view<@_ZTIM6Recordi> : !cir.ptr<!u8i>] (%{{.*}}: !cir.eh_token {{.*}} {
// CIR:     %[[CATCH_TOKEN:.*]], %[[EXN_PTR:.*]] = cir.begin_catch %{{.*}} : !cir.eh_token -> (!cir.catch_token, !cir.ptr<!void>)
// CIR:     cir.cleanup.scope {
// CIR:       cir.init_catch_param scalar %[[EXN_PTR]] to %{{.*}} : !cir.ptr<!void>, !cir.ptr<!s64i>
// CIR:       cir.yield
// CIR:     } cleanup all {
// CIR:       cir.end_catch %[[CATCH_TOKEN]] : !cir.catch_token
// CIR:       cir.yield
// CIR:     }
// CIR:     cir.yield
// CIR:   } unwind (%{{.*}}: !cir.eh_token {{.*}} {
// CIR:     cir.resume %{{.*}} : !cir.eh_token
// CIR:   }
// CIR: }

// LLVM: define {{.*}} void @_Z61call_function_inside_try_catch_with_exception_member_ptr_typev() {{.*}} personality ptr @__gxx_personality_v0
// OGCG:   %[[EXN_SLOT:.*]] = alloca ptr, align 8
// OGCG:   %[[EH_SELECTOR_SLOT:.*]] = alloca i32, align 4
// LLVM:   %[[E_ADDR:.*]] = alloca i64, align 8
// LLVMCIR:   br label %[[TRY_SCOPE:.*]]
// LLVMCIR: [[TRY_SCOPE]]:
// LLVMCIR:   br label %[[TRY_BEGIN:.*]]
// LLVMCIR: [[TRY_BEGIN]]:
// LLVM:   %[[CALL:.*]] = invoke noundef i32 @_Z8divisionv()
// LLVM:           to label %[[INVOKE_CONT:.*]] unwind label %[[LANDING_PAD:.*]]
// LLVM: [[INVOKE_CONT]]:
// LLVM:   br label %[[TRY_CONT:.*]]
// LLVM: [[LANDING_PAD]]:
// LLVM:   %[[LP:.*]] = landingpad { ptr, i32 }
// LLVM:           catch ptr @_ZTIM6Recordi
// LLVM:   %[[EXN_OBJ:.*]] = extractvalue { ptr, i32 } %[[LP]], 0
// OGCG:   store ptr %[[EXN_OBJ]], ptr %[[EXN_SLOT]], align 8
// LLVM:   %[[EH_SELECTOR_VAL:.*]] = extractvalue { ptr, i32 } %[[LP]], 1
// OGCG:   store i32 %[[EH_SELECTOR_VAL]], ptr %[[EH_SELECTOR_SLOT]], align 4
// LLVM:   br label %[[CATCH:.*]]
// LLVM: [[CATCH]]:
// LLVMCIR:   %[[EXN_OBJ_PHI:.*]] = phi ptr [ %[[EXN_OBJ]], %[[LANDING_PAD]] ]
// LLVMCIR:   %[[EH_SELECTOR_PHI:.*]] = phi i32 [ %[[EH_SELECTOR_VAL]], %[[LANDING_PAD]] ]
// LLVMCIR:   br label %[[DISPATCH:.*]]
// LLVMCIR: [[DISPATCH]]:
// LLVMCIR:   %[[DISPATCH_EXN:.*]] = phi ptr [ %[[EXN_OBJ_PHI]], %[[CATCH]] ]
// LLVMCIR:   %[[EH_SELECTOR:.*]] = phi i32 [ %[[EH_SELECTOR_PHI]], %[[CATCH]] ]
// OGCG:   %[[EH_SELECTOR:.*]] = load i32, ptr %[[EH_SELECTOR_SLOT]], align 4
// LLVM:   %[[EH_TYPE_ID:.*]] = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIM6Recordi)
// LLVM:   %[[TYPE_ID_EQ:.*]] = icmp eq i32 %[[EH_SELECTOR]], %[[EH_TYPE_ID]]
// LLVM:   br i1 %[[TYPE_ID_EQ]], label %[[BEGIN_CATCH:.*]], label %[[RESUME:.*]]
// LLVM: [[BEGIN_CATCH]]:
// LLVMCIR:   %[[EXN:.*]] = phi ptr [ %[[DISPATCH_EXN]], %[[DISPATCH]] ]
// LLVMCIR:   %{{.*}} = phi i32 [ %[[EH_SELECTOR]], %[[DISPATCH]] ]
// OGCG:   %[[EXN:.*]] = load ptr, ptr %[[EXN_SLOT]], align 8
// LLVM:   %[[EXN_PTR:.*]] = call ptr @__cxa_begin_catch(ptr %[[EXN]])
// LLVMCIR:   br label %[[CATCH_BODY:.*]]
// LLVMCIR: [[CATCH_BODY]]:
// LLVM:   %[[TMP_EXN:.*]] = load i64, ptr %[[EXN_PTR]], align 8
// LLVM:   store i64 %[[TMP_EXN]], ptr %[[E_ADDR]], align 8
// LLVMCIR:   br label %[[END_CATCH:.*]]
// LLVMCIR: [[END_CATCH]]:
// LLVM:   call void @__cxa_end_catch()
// LLVMCIR:   br label %[[END_DISPATCH:.*]]
// LLVMCIR: [[END_DISPATCH]]:
// LLVMCIR:   br label %[[END_TRY:.*]]
// LLVMCIR: [[END_TRY]]:
// LLVM:   br label %[[TRY_CONT]]
// OGCG: [[TRY_CONT]]:
// OGCG:   ret void
// LLVM: [[RESUME]]:
// LLVMCIR:   %[[RESUME_EXN:.*]] = phi ptr [ %[[DISPATCH_EXN]], %[[DISPATCH]] ]
// LLVMCIR:   %[[RESUME_SEL:.*]] = phi i32 [ %[[EH_SELECTOR]], %[[DISPATCH]] ]
// OGCG:   %[[RESUME_EXN:.*]] = load ptr, ptr %[[EXN_SLOT]], align 8
// OGCG:   %[[RESUME_SEL:.*]] = load i32, ptr %[[EH_SELECTOR_SLOT]], align 4
// LLVM:   %[[TMP_EXCEPTION_INFO:.*]] = insertvalue { ptr, i32 } poison, ptr %[[RESUME_EXN]], 0
// LLVM:   %[[EXCEPTION_INFO:.*]] = insertvalue { ptr, i32 } %[[TMP_EXCEPTION_INFO]], i32 %[[RESUME_SEL]], 1
// LLVM:   resume { ptr, i32 } %[[EXCEPTION_INFO]]
// LLVMCIR: [[TRY_CONT]]:
// LLVMCIR:   br label %[[DONE:.*]]
// LLVMCIR: [[DONE]]:
// LLVMCIR:   ret void

int init_catch_param_with_type_int() {
  int rv = 0;
  try {
    division();
  } catch (int x) {
    rv = x;
  }
  return rv;
}

// CIR: cir.func {{.*}} @_Z30init_catch_param_with_type_intv() {{.*}} personality(@__gxx_personality_v0)
// CIR:   %[[RET_ADDR:.*]] = cir.alloca "__retval" {{.*}} : !cir.ptr<!s32i>
// CIR:   %[[RV_ADDR:.*]] = cir.alloca "rv" {{.*}} init : !cir.ptr<!s32i>
// CIR:   cir.scope {
// CIR:     %[[X_ADDR:.*]] = cir.alloca "x" {{.*}} : !cir.ptr<!s32i>
// CIR:     cir.try {
// CIR:       %[[CALL:.*]] = cir.call @_Z8divisionv() : () -> (!s32i {llvm.noundef})
// CIR:       cir.yield
// CIR:     } catch [type #cir.global_view<@_ZTIi> : !cir.ptr<!u8i>] (%{{.*}}: !cir.eh_token {{.*}}) {
// CIR:       %[[CATCH_TOKEN:.*]], %[[EXN_PTR:.*]] = cir.begin_catch %{{.*}} : !cir.eh_token -> (!cir.catch_token, !cir.ptr<!void>)
// CIR:       cir.cleanup.scope {
// CIR:         cir.init_catch_param scalar %[[EXN_PTR]] to %[[X_ADDR]] : !cir.ptr<!void>, !cir.ptr<!s32i>
// CIR:         %[[TMP_EXN:.*]] = cir.load {{.*}} %[[X_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:         cir.store {{.*}} %[[TMP_EXN]], %[[RV_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:         cir.yield
// CIR:       } cleanup all {
// CIR:         cir.end_catch %[[CATCH_TOKEN]] : !cir.catch_token
// CIR:         cir.yield
// CIR:       }
// CIR:       cir.yield
// CIR:     } unwind (%{{.*}}: !cir.eh_token {{.*}}) {
// CIR:       cir.resume %{{.*}} : !cir.eh_token
// CIR:     }
// CIR:   }
// CIR:   %[[TMP_RV:.*]] = cir.load {{.*}} %[[RV_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.store %[[TMP_RV]], %[[RET_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[TMP_RET:.*]] = cir.load %[[RET_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.return %[[TMP_RET]] : !s32i

// LLVM: define {{.*}} i32 @_Z30init_catch_param_with_type_intv() {{.*}} personality ptr @__gxx_personality_v0
// LLVMCIR:   %[[X_ADDR:.*]] = alloca i32, align 4
// LLVMCIR:   %[[RET_ADDR:.*]] = alloca i32, align 4
// LLVM:   %[[RV_ADDR:.*]] = alloca i32, align 4
// OGCG:   %[[EXN_SLOT:.*]] = alloca ptr, align 8
// OGCG:   %[[EH_SELECTOR_SLOT:.*]] = alloca i32, align 4
// OGCG:   %[[X_ADDR:.*]] = alloca i32, align 4
// LLVM:   store i32 0, ptr %[[RV_ADDR]], align 4
// LLVMCIR:   br label %[[TRY_SCOPE:.*]]
// LLVMCIR: [[TRY_SCOPE]]:
// LLVMCIR:   br label %[[TRY_BEGIN:.*]]
// LLVMCIR: [[TRY_BEGIN]]:
// LLVM:   %[[CALL:.*]] = invoke noundef i32 @_Z8divisionv()
// LLVM:           to label %[[INVOKE_CONT:.*]] unwind label %[[LANDING_PAD:.*]]
// LLVM: [[INVOKE_CONT]]:
// LLVM:   br label %[[TRY_CONT:.*]]
// LLVM: [[LANDING_PAD]]:
// LLVM:   %[[LP:.*]] = landingpad { ptr, i32 }
// LLVM:           catch ptr @_ZTIi
// LLVM:   %[[EXN_OBJ:.*]] = extractvalue { ptr, i32 } %[[LP]], 0
// OGCG:   store ptr %[[EXN_OBJ]], ptr %[[EXN_SLOT]], align 8
// LLVM:   %[[EH_SELECTOR_VAL:.*]] = extractvalue { ptr, i32 } %[[LP]], 1
// OGCG:   store i32 %[[EH_SELECTOR_VAL]], ptr %[[EH_SELECTOR_SLOT]], align 4
// LLVM:   br label %[[CATCH:.*]]
// LLVM: [[CATCH]]:
// LLVMCIR:   %[[EXN_OBJ_PHI:.*]] = phi ptr [ %[[EXN_OBJ]], %[[LANDING_PAD]] ]
// LLVMCIR:   %[[EH_SELECTOR_PHI:.*]] = phi i32 [ %[[EH_SELECTOR_VAL]], %[[LANDING_PAD]] ]
// LLVMCIR:   br label %[[DISPATCH:.*]]
// LLVMCIR: [[DISPATCH]]:
// LLVMCIR:   %[[DISPATCH_EXN:.*]] = phi ptr [ %[[EXN_OBJ_PHI]], %[[CATCH]] ]
// LLVMCIR:   %[[EH_SELECTOR:.*]] = phi i32 [ %[[EH_SELECTOR_PHI]], %[[CATCH]] ]
// OGCG:   %[[EH_SELECTOR:.*]] = load i32, ptr %[[EH_SELECTOR_SLOT]], align 4
// LLVM:   %[[EH_TYPE_ID:.*]] = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
// LLVM:   %[[TYPE_ID_EQ:.*]] = icmp eq i32 %[[EH_SELECTOR]], %[[EH_TYPE_ID]]
// LLVM:   br i1 %[[TYPE_ID_EQ]], label %[[BEGIN_CATCH:.*]], label %[[RESUME:.*]]
// LLVM: [[BEGIN_CATCH]]:
// LLVMCIR:   %[[EXN:.*]] = phi ptr [ %[[DISPATCH_EXN]], %[[DISPATCH]] ]
// LLVMCIR:   %{{.*}} = phi i32 [ %[[EH_SELECTOR]], %[[DISPATCH]] ]
// OGCG:   %[[EXN:.*]] = load ptr, ptr %[[EXN_SLOT]], align 8
// LLVM:   %[[EXN_PTR:.*]] = call ptr @__cxa_begin_catch(ptr %[[EXN]])
// LLVMCIR:   br label %[[CATCH_BODY:.*]]
// LLVMCIR: [[CATCH_BODY]]:
// LLVM:   %[[TMP_EXN:.*]] = load i32, ptr %[[EXN_PTR]], align 4
// LLVM:   store i32 %[[TMP_EXN]], ptr %[[X_ADDR]], align 4
// LLVM:   %[[TMP_X:.*]] = load i32, ptr %[[X_ADDR]], align 4
// LLVM:   store i32 %[[TMP_X]], ptr %[[RV_ADDR]], align 4
// LLVMCIR:   br label %[[END_CATCH:.*]]
// LLVMCIR: [[END_CATCH]]:
// LLVM:   call void @__cxa_end_catch()
// LLVMCIR:   br label %[[END_DISPATCH:.*]]
// LLVMCIR: [[END_DISPATCH]]:
// LLVMCIR:   br label %[[END_TRY:.*]]
// LLVMCIR: [[END_TRY]]:
// LLVM:   br label %[[TRY_CONT]]
// OGCG: [[TRY_CONT]]:
// OGCG:   %[[TMP_RV:.*]] = load i32, ptr %[[RV_ADDR]], align 4
// OGCG:   ret i32 %[[TMP_RV]]
// LLVM: [[RESUME]]:
// LLVMCIR:   %[[RESUME_EXN:.*]] = phi ptr [ %[[DISPATCH_EXN]], %[[DISPATCH]] ]
// LLVMCIR:   %[[RESUME_SEL:.*]] = phi i32 [ %[[EH_SELECTOR]], %[[DISPATCH]] ]
// OGCG:   %[[RESUME_EXN:.*]] = load ptr, ptr %[[EXN_SLOT]], align 8
// OGCG:   %[[RESUME_SEL:.*]] = load i32, ptr %[[EH_SELECTOR_SLOT]], align 4
// LLVM:   %[[TMP_EXCEPTION_INFO:.*]] = insertvalue { ptr, i32 } poison, ptr %[[RESUME_EXN]], 0
// LLVM:   %[[EXCEPTION_INFO:.*]] = insertvalue { ptr, i32 } %[[TMP_EXCEPTION_INFO]], i32 %[[RESUME_SEL]], 1
// LLVM:   resume { ptr, i32 } %[[EXCEPTION_INFO]]
// LLVMCIR: [[TRY_CONT]]:
// LLVMCIR:   br label %[[DONE:.*]]
// LLVMCIR: [[DONE]]:
// LLVMCIR:   %[[TMP_RV:.*]] = load i32, ptr %[[RV_ADDR]], align 4
// LLVMCIR:   store i32 %[[TMP_RV]], ptr %[[RET_ADDR]], align 4
// LLVMCIR:   %[[TMP_RET:.*]] = load i32, ptr %[[RET_ADDR]], align 4
// LLVMCIR:   ret i32 %[[TMP_RET]]


int init_catch_param_with_type_int_ptr() {
  int rv = 0;
  try {
    division();
  } catch (int *x) {
    rv = *x;
  }
  return rv;
}

// CIR: cir.func {{.*}} @_Z34init_catch_param_with_type_int_ptrv() {{.*}} personality(@__gxx_personality_v0)
// CIR:   %[[RET_ADDR:.*]] = cir.alloca "__retval" {{.*}} : !cir.ptr<!s32i>
// CIR:   %[[RV_ADDR:.*]] = cir.alloca "rv" {{.*}} init : !cir.ptr<!s32i>
// CIR:   cir.scope {
// CIR:     %[[X_ADDR:.*]] = cir.alloca "x" {{.*}} : !cir.ptr<!cir.ptr<!s32i>>
// CIR:     cir.try {
// CIR:       %[[CALL:.*]] = cir.call @_Z8divisionv() : () -> (!s32i {llvm.noundef})
// CIR:       cir.yield
// CIR:     } catch [type #cir.global_view<@_ZTIPi> : !cir.ptr<!u8i>] (%{{.*}}: !cir.eh_token {{.*}}) {
// CIR:       %[[CATCH_TOKEN:.*]], %[[EXN_PTR:.*]] = cir.begin_catch %{{.*}} : !cir.eh_token -> (!cir.catch_token, !cir.ptr<!void>)
// CIR:       cir.cleanup.scope {
// CIR:         cir.init_catch_param pointer %[[EXN_PTR]] to %[[X_ADDR]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!s32i>>
// CIR:         %[[DEREF_X:.*]] = cir.load deref {{.*}} %[[X_ADDR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR:         %[[LOADED_INT:.*]] = cir.load {{.*}} %[[DEREF_X]] : !cir.ptr<!s32i>, !s32i
// CIR:         cir.store {{.*}} %[[LOADED_INT]], %[[RV_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:         cir.yield
// CIR:       } cleanup all {
// CIR:         cir.end_catch %[[CATCH_TOKEN]] : !cir.catch_token
// CIR:         cir.yield
// CIR:       }
// CIR:       cir.yield
// CIR:     } unwind (%{{.*}}: !cir.eh_token {{.*}} {
// CIR:       cir.resume %{{.*}} : !cir.eh_token
// CIR:     }
// CIR:   }
// CIR:   %[[TMP_RV:.*]] = cir.load {{.*}} %[[RV_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.store %[[TMP_RV]], %[[RET_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[TMP_RET:.*]] = cir.load %[[RET_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.return %[[TMP_RET]] : !s32i

// LLVM: define {{.*}} i32 @_Z34init_catch_param_with_type_int_ptrv() {{.*}} personality ptr @__gxx_personality_v0
// LLVMCIR:   %[[X_ADDR:.*]] = alloca ptr, align 8
// LLVMCIR:   %[[RET_ADDR:.*]] = alloca i32, align 4
// LLVM:   %[[RV_ADDR:.*]] = alloca i32, align 4
// OGCG:   %[[EXN_SLOT:.*]] = alloca ptr, align 8
// OGCG:   %[[EH_SELECTOR_SLOT:.*]] = alloca i32, align 4
// OGCG:   %[[X_ADDR:.*]] = alloca ptr, align 8
// LLVM:   store i32 0, ptr %[[RV_ADDR]], align 4
// LLVMCIR:   br label %[[TRY_SCOPE:.*]]
// LLVMCIR: [[TRY_SCOPE]]:
// LLVMCIR:   br label %[[TRY_BEGIN:.*]]
// LLVMCIR: [[TRY_BEGIN]]:
// LLVM:   %[[CALL:.*]] = invoke noundef i32 @_Z8divisionv()
// LLVM:           to label %[[INVOKE_CONT:.*]] unwind label %[[LANDING_PAD:.*]]
// LLVM: [[INVOKE_CONT]]:
// LLVM:   br label %[[TRY_CONT:.*]]
// LLVM: [[LANDING_PAD]]:
// LLVM:   %[[LP:.*]] = landingpad { ptr, i32 }
// LLVM:           catch ptr @_ZTIPi
// LLVM:   %[[EXN_OBJ:.*]] = extractvalue { ptr, i32 } %[[LP]], 0
// OGCG:   store ptr %[[EXN_OBJ]], ptr %[[EXN_SLOT]], align 8
// LLVM:   %[[EH_SELECTOR_VAL:.*]] = extractvalue { ptr, i32 } %[[LP]], 1
// OGCG:   store i32 %[[EH_SELECTOR_VAL]], ptr %[[EH_SELECTOR_SLOT]], align 4
// LLVM:   br label %[[CATCH:.*]]
// LLVM: [[CATCH]]:
// LLVMCIR:   %[[EXN_OBJ_PHI:.*]] = phi ptr [ %[[EXN_OBJ]], %[[LANDING_PAD]] ]
// LLVMCIR:   %[[EH_SELECTOR_PHI:.*]] = phi i32 [ %[[EH_SELECTOR_VAL]], %[[LANDING_PAD]] ]
// LLVMCIR:   br label %[[DISPATCH:.*]]
// LLVMCIR: [[DISPATCH]]:
// LLVMCIR:   %[[DISPATCH_EXN:.*]] = phi ptr [ %[[EXN_OBJ_PHI]], %[[CATCH]] ]
// LLVMCIR:   %[[EH_SELECTOR:.*]] = phi i32 [ %[[EH_SELECTOR_PHI]], %[[CATCH]] ]
// OGCG:   %[[EH_SELECTOR:.*]] = load i32, ptr %[[EH_SELECTOR_SLOT]], align 4
// LLVM:   %[[EH_TYPE_ID:.*]] = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIPi)
// LLVM:   %[[TYPE_ID_EQ:.*]] = icmp eq i32 %[[EH_SELECTOR]], %[[EH_TYPE_ID]]
// LLVM:   br i1 %[[TYPE_ID_EQ]], label %[[BEGIN_CATCH:.*]], label %[[RESUME:.*]]
// LLVM: [[BEGIN_CATCH]]:
// LLVMCIR:   %[[EXN:.*]] = phi ptr [ %[[DISPATCH_EXN]], %[[DISPATCH]] ]
// LLVMCIR:   %{{.*}} = phi i32 [ %[[EH_SELECTOR]], %[[DISPATCH]] ]
// OGCG:   %[[EXN:.*]] = load ptr, ptr %[[EXN_SLOT]], align 8
// LLVM:   %[[EXN_PTR:.*]] = call ptr @__cxa_begin_catch(ptr %[[EXN]])
// LLVMCIR:   br label %[[CATCH_BODY:.*]]
// LLVMCIR: [[CATCH_BODY]]:
// LLVM:   store ptr %[[EXN_PTR]], ptr %[[X_ADDR]], align 8
// LLVM:   %[[DEREF_X:.*]] = load ptr, ptr %[[X_ADDR]], align 8
// LLVM:   %[[TMP_X:.*]] = load i32, ptr %[[DEREF_X]], align 4
// LLVM:   store i32 %[[TMP_X]], ptr %[[RV_ADDR]], align 4
// LLVMCIR:   br label %[[END_CATCH:.*]]
// LLVMCIR: [[END_CATCH]]:
// LLVM:   call void @__cxa_end_catch()
// LLVMCIR:   br label %[[END_DISPATCH:.*]]
// LLVMCIR: [[END_DISPATCH]]:
// LLVMCIR:   br label %[[END_TRY:.*]]
// LLVMCIR: [[END_TRY]]:
// LLVM:   br label %[[TRY_CONT]]
// OGCG: [[TRY_CONT]]:
// OGCG:   %[[TMP_RV:.*]] = load i32, ptr %[[RV_ADDR]], align 4
// OGCG:   ret i32 %[[TMP_RV]]
// LLVM: [[RESUME]]:
// LLVMCIR:   %[[RESUME_EXN:.*]] = phi ptr [ %[[DISPATCH_EXN]], %[[DISPATCH]] ]
// LLVMCIR:   %[[RESUME_SEL:.*]] = phi i32 [ %[[EH_SELECTOR]], %[[DISPATCH]] ]
// OGCG:   %[[RESUME_EXN:.*]] = load ptr, ptr %[[EXN_SLOT]], align 8
// OGCG:   %[[RESUME_SEL:.*]] = load i32, ptr %[[EH_SELECTOR_SLOT]], align 4
// LLVM:   %[[TMP_EXCEPTION_INFO:.*]] = insertvalue { ptr, i32 } poison, ptr %[[RESUME_EXN]], 0
// LLVM:   %[[EXCEPTION_INFO:.*]] = insertvalue { ptr, i32 } %[[TMP_EXCEPTION_INFO]], i32 %[[RESUME_SEL]], 1
// LLVM:   resume { ptr, i32 } %[[EXCEPTION_INFO]]
// LLVMCIR: [[TRY_CONT]]:
// LLVMCIR:   br label %[[DONE:.*]]
// LLVMCIR: [[DONE]]:
// LLVMCIR:   %[[TMP_RV:.*]] = load i32, ptr %[[RV_ADDR]], align 4
// LLVMCIR:   store i32 %[[TMP_RV]], ptr %[[RET_ADDR]], align 4
// LLVMCIR:   %[[TMP_RET:.*]] = load i32, ptr %[[RET_ADDR]], align 4
// LLVMCIR:   ret i32 %[[TMP_RET]]

int init_catch_param_with_ref_to_ptr_to_non_record() {
  int rv = 0;
  try {
    division();
  } catch (int *&p) {
    rv = *p;
  }
  return rv;
}

// CIR: cir.func {{.*}} @_Z46init_catch_param_with_ref_to_ptr_to_non_recordv() {{.*}} personality(@__gxx_personality_v0)
// CIR:   %[[RET_ADDR:.*]] = cir.alloca "__retval" {{.*}} : !cir.ptr<!s32i>
// CIR:   %[[RV_ADDR:.*]] = cir.alloca "rv" {{.*}} init : !cir.ptr<!s32i>
// CIR:   cir.scope {
// CIR:     %[[P_ADDR:.*]] = cir.alloca "p" {{.*}} const : !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>
// CIR:     cir.try {
// CIR:       %[[CALL:.*]] = cir.call @_Z8divisionv() : () -> (!s32i {llvm.noundef})
// CIR:       cir.yield
// CIR:     } catch [type #cir.global_view<@_ZTIPi> : !cir.ptr<!u8i>] (%[[TOKEN:.*]]: !cir.eh_token {{.*}}) {
// CIR:       cir.construct_catch_param reference_to_pointer %[[TOKEN]] to %[[P_ADDR]] using : !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>
// CIR:       %[[CATCH_TOKEN:.*]], %[[EXN_PTR:.*]] = cir.begin_catch %{{.*}} : !cir.eh_token -> (!cir.catch_token, !cir.ptr<!void>)
// CIR:       cir.cleanup.scope {
// CIR:         cir.init_catch_param reference_to_pointer %[[EXN_PTR]] to %[[P_ADDR]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>
// CIR:         %[[TMP_P:.*]] = cir.load %[[P_ADDR]] : !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>, !cir.ptr<!cir.ptr<!s32i>>
// CIR:         %[[DEREF_P:.*]] = cir.load deref {{.*}} %[[TMP_P]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR:         %[[P_VAL:.*]] = cir.load {{.*}} %[[DEREF_P]] : !cir.ptr<!s32i>, !s32i
// CIR:         cir.store {{.*}} %[[P_VAL]], %[[RV_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:         cir.yield
// CIR:       } cleanup all {
// CIR:         cir.end_catch %[[CATCH_TOKEN]] : !cir.catch_token
// CIR:         cir.yield
// CIR:       }
// CIR:       cir.yield
// CIR:     } unwind (%{{.*}}: !cir.eh_token {{.*}}) {
// CIR:       cir.resume %{{.*}} : !cir.eh_token
// CIR:     }
// CIR:   }
// CIR:   %[[TMP_RV:.*]] = cir.load {{.*}} %[[RV_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.store %[[TMP_RV]], %[[RET_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[TMP_RET:.*]] = cir.load %[[RET_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.return %[[TMP_RET]] : !s32i

// LLVM: define {{.*}} i32 @_Z46init_catch_param_with_ref_to_ptr_to_non_recordv() {{.*}} personality ptr @__gxx_personality_v0
// LLVMCIR:   %[[P_ADDR:.*]] = alloca ptr, align 8
// LLVMCIR:   %[[RET_ADDR:.*]] = alloca i32, align 4
// LLVM:   %[[RV_ADDR:.*]] = alloca i32, align 4
// OGCG:   %[[EXN_SLOT:.*]] = alloca ptr, align 8
// OGCG:   %[[EH_SELECTOR_SLOT:.*]] = alloca i32, align 4
// OGCG:   %[[P_ADDR:.*]] = alloca ptr, align 8
// LLVM:   store i32 0, ptr %[[RV_ADDR]], align 4
// LLVMCIR:   br label %[[TRY_SCOPE:.*]]
// LLVMCIR: [[TRY_SCOPE]]:
// LLVMCIR:   br label %[[TRY_BEGIN:.*]]
// LLVMCIR: [[TRY_BEGIN]]:
// LLVM:   %[[CALL:.*]] = invoke noundef i32 @_Z8divisionv()
// LLVM:           to label %[[INVOKE_CONT:.*]] unwind label %[[LANDING_PAD:.*]]
// LLVM: [[INVOKE_CONT]]:
// LLVM:   br label %[[TRY_CONT:.*]]
// LLVM: [[LANDING_PAD]]:
// LLVM:   %[[LP:.*]] = landingpad { ptr, i32 }
// LLVM:           catch ptr @_ZTIPi
// LLVM:   %[[EXN_OBJ:.*]] = extractvalue { ptr, i32 } %[[LP]], 0
// OGCG:   store ptr %[[EXN_OBJ]], ptr %[[EXN_SLOT]], align 8
// LLVM:   %[[EH_SELECTOR_VAL:.*]] = extractvalue { ptr, i32 } %[[LP]], 1
// OGCG:   store i32 %[[EH_SELECTOR_VAL]], ptr %[[EH_SELECTOR_SLOT]], align 4
// LLVM:   br label %[[CATCH:.*]]
// LLVM: [[CATCH]]:
// LLVMCIR:   %[[EXN_OBJ_PHI:.*]] = phi ptr [ %[[EXN_OBJ]], %[[LANDING_PAD]] ]
// LLVMCIR:   %[[EH_SELECTOR_PHI:.*]] = phi i32 [ %[[EH_SELECTOR_VAL]], %[[LANDING_PAD]] ]
// LLVMCIR:   br label %[[DISPATCH:.*]]
// LLVMCIR: [[DISPATCH]]:
// LLVMCIR:   %[[DISPATCH_EXN:.*]] = phi ptr [ %[[EXN_OBJ_PHI]], %[[CATCH]] ]
// LLVMCIR:   %[[EH_SELECTOR:.*]] = phi i32 [ %[[EH_SELECTOR_PHI]], %[[CATCH]] ]
// OGCG:   %[[EH_SELECTOR:.*]] = load i32, ptr %[[EH_SELECTOR_SLOT]], align 4
// LLVM:   %[[EH_TYPE_ID:.*]] = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIPi)
// LLVM:   %[[TYPE_ID_EQ:.*]] = icmp eq i32 %[[EH_SELECTOR]], %[[EH_TYPE_ID]]
// LLVM:   br i1 %[[TYPE_ID_EQ]], label %[[BEGIN_CATCH:.*]], label %[[RESUME:.*]]
// LLVM: [[BEGIN_CATCH]]:
// LLVMCIR:   %[[EXN:.*]] = phi ptr [ %[[DISPATCH_EXN]], %[[DISPATCH]] ]
// LLVMCIR:   %{{.*}} = phi i32 [ %[[EH_SELECTOR]], %[[DISPATCH]] ]
// LLVMCIR:   %[[ADJUSTED_EXN:.*]] = getelementptr i8, ptr %[[EXN]], i64 32
// LLVMCIR:   store ptr %[[ADJUSTED_EXN]], ptr %[[P_ADDR]], align 8
// OGCG:   %[[EXN:.*]] = load ptr, ptr %[[EXN_SLOT]], align 8
// LLVM:   %[[EXN_PTR:.*]] = call ptr @__cxa_begin_catch(ptr %[[EXN]])
// LLVMCIR:   br label %[[CATCH_BODY:.*]]
// LLVMCIR: [[CATCH_BODY]]:
// OGCG:   %[[ADJUSTED_EXN:.*]] = getelementptr i8, ptr %[[EXN]], i32 32
// OGCG:   store ptr %[[ADJUSTED_EXN]], ptr %[[P_ADDR]], align 8
// LLVM:   %[[TMP_P:.*]] = load ptr, ptr %[[P_ADDR]], align 8
// LLVM:   %[[DEREF_P:.*]] = load ptr, ptr %[[TMP_P]], align 8
// LLVM:   %[[P_VAL:.*]] = load i32, ptr %[[DEREF_P]], align 4
// LLVM:   store i32 %[[P_VAL]], ptr %[[RV_ADDR]], align 4
// LLVMCIR:   br label %[[END_CATCH:.*]]
// LLVMCIR: [[END_CATCH]]:
// LLVM:   call void @__cxa_end_catch()
// LLVMCIR:   br label %[[END_DISPATCH:.*]]
// LLVMCIR: [[END_DISPATCH]]:
// LLVMCIR:   br label %[[END_TRY:.*]]
// LLVMCIR: [[END_TRY]]:
// LLVM:   br label %[[TRY_CONT]]
// OGCG: [[TRY_CONT]]:
// OGCG:   %[[TMP_RV:.*]] = load i32, ptr %[[RV_ADDR]], align 4
// OGCG:   ret i32 %[[TMP_RV]]
// LLVM: [[RESUME]]:
// LLVMCIR:   %[[RESUME_EXN:.*]] = phi ptr [ %[[DISPATCH_EXN]], %[[DISPATCH]] ]
// LLVMCIR:   %[[RESUME_SEL:.*]] = phi i32 [ %[[EH_SELECTOR]], %[[DISPATCH]] ]
// OGCG:   %[[RESUME_EXN:.*]] = load ptr, ptr %[[EXN_SLOT]], align 8
// OGCG:   %[[RESUME_SEL:.*]] = load i32, ptr %[[EH_SELECTOR_SLOT]], align 4
// LLVM:   %[[TMP_EXCEPTION_INFO:.*]] = insertvalue { ptr, i32 } poison, ptr %[[RESUME_EXN]], 0
// LLVM:   %[[EXCEPTION_INFO:.*]] = insertvalue { ptr, i32 } %[[TMP_EXCEPTION_INFO]], i32 %[[RESUME_SEL]], 1
// LLVM:   resume { ptr, i32 } %[[EXCEPTION_INFO]]
// LLVMCIR: [[TRY_CONT]]:
// LLVMCIR:   br label %[[DONE:.*]]
// LLVMCIR: [[DONE]]:
// LLVMCIR:   %[[TMP_RV:.*]] = load i32, ptr %[[RV_ADDR]], align 4
// LLVMCIR:   store i32 %[[TMP_RV]], ptr %[[RET_ADDR]], align 4
// LLVMCIR:   %[[TMP_RET:.*]] = load i32, ptr %[[RET_ADDR]], align 4
// LLVMCIR:   ret i32 %[[TMP_RET]]

void direct_inside_try_catch_with_exception_type() {
  try {
    throw 42;
  } catch (int e) {
  }
}

// CIR: cir.func {{.*}} @_Z43direct_inside_try_catch_with_exception_typev() personality(@__gxx_personality_v0)
// CIR:   cir.scope {
// CIR:     %[[E:.*]] = cir.alloca "e" {{.*}} : !cir.ptr<!s32i>
// CIR:     cir.try {
// CIR:       %[[EXN:.*]] = cir.alloc.exception 4 -> !cir.ptr<!s32i>
// CIR:       %[[FORTYTWO:.*]] = cir.const #cir.int<42> : !s32i
// CIR:       cir.store{{.*}} %[[FORTYTWO]], %[[EXN]]
// CIR:       cir.throw %[[EXN]] : !cir.ptr<!s32i>, @_ZTIi
// CIR:       cir.unreachable
// CIR:     } catch [type #cir.global_view<@_ZTIi> : !cir.ptr<!u8i>] (%[[TOKEN:.*]]: !cir.eh_token {{.*}}) {
// CIR:       %[[CATCH_TOKEN:.*]], %[[EXN_PTR:.*]] = cir.begin_catch %{{.*}} : !cir.eh_token -> (!cir.catch_token, !cir.ptr<!void>)
// CIR:       cir.cleanup.scope {
// CIR:         cir.init_catch_param scalar %[[EXN_PTR]] to %[[E]] : !cir.ptr<!void>, !cir.ptr<!s32i>
// CIR:         cir.yield
// CIR:       } cleanup all {
// CIR:         cir.end_catch %[[CATCH_TOKEN]] : !cir.catch_token
// CIR:         cir.yield
// CIR:       }
// CIR:       cir.yield
// CIR:     } unwind (%{{.*}}: !cir.eh_token {{.*}}) {
// CIR:       cir.resume %{{.*}} : !cir.eh_token
// CIR:     }
// CIR:   }

// LLVM: define {{.*}} void @_Z43direct_inside_try_catch_with_exception_typev() {{.*}} personality ptr @__gxx_personality_v0 {
// OGCG:   %[[EXN_SLOT:.*]] = alloca ptr, align 8
// OGCG:   %[[EH_SELECTOR_SLOT:.*]] = alloca i32, align 4
// LLVM:   %[[E:.*]] = alloca i32, align 4
// LLVMCIR:   br label %[[TRY_SCOPE:.*]]
// LLVMCIR: [[TRY_SCOPE]]:
// LLVMCIR:   br label %[[TRY_BEGIN:.*]]
// LLVMCIR: [[TRY_BEGIN]]:
// LLVM:   %[[THROWN_EXN:.*]] = call ptr @__cxa_allocate_exception(i64 4)
// LLVM:   store i32 42, ptr %[[THROWN_EXN]], align 16
// LLVM:   invoke void @__cxa_throw(ptr %[[THROWN_EXN]], ptr @_ZTIi, ptr null)
// LLVM:           to label %[[UNREACHABLE:.*]] unwind label %[[LANDING_PAD:.*]]
// LLVMCIR: [[UNREACHABLE]]:
// LLVMCIR:   unreachable
// LLVM: [[LANDING_PAD]]:
// LLVM:   %[[LP:.*]] = landingpad { ptr, i32 }
// LLVM:           catch ptr @_ZTIi
// LLVM:   %[[EXN_OBJ:.*]] = extractvalue { ptr, i32 } %[[LP]], 0
// OGCG:   store ptr %[[EXN_OBJ]], ptr %[[EXN_SLOT]], align 8
// LLVM:   %[[EH_SELECTOR_VAL:.*]] = extractvalue { ptr, i32 } %[[LP]], 1
// OGCG:   store i32 %[[EH_SELECTOR_VAL]], ptr %[[EH_SELECTOR_SLOT]], align 4
// LLVM:   br label %[[CATCH:.*]]
// LLVM: [[CATCH]]:
// LLVMCIR:   %[[EXN_OBJ_PHI:.*]] = phi ptr [ %[[EXN_OBJ]], %[[LANDING_PAD]] ]
// LLVMCIR:   %[[EH_SELECTOR_PHI:.*]] = phi i32 [ %[[EH_SELECTOR_VAL]], %[[LANDING_PAD]] ]
// LLVMCIR:   br label %[[DISPATCH:.*]]
// LLVMCIR: [[DISPATCH]]:
// LLVMCIR:   %[[DISPATCH_EXN:.*]] = phi ptr [ %[[EXN_OBJ_PHI]], %[[CATCH]] ]
// LLVMCIR:   %[[EH_SELECTOR:.*]] = phi i32 [ %[[EH_SELECTOR_PHI]], %[[CATCH]] ]
// OGCG:   %[[EH_SELECTOR:.*]] = load i32, ptr %[[EH_SELECTOR_SLOT]], align 4
// LLVM:   %[[EH_TYPE_ID:.*]] = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
// LLVM:   %[[TYPE_ID_EQ:.*]] = icmp eq i32 %[[EH_SELECTOR]], %[[EH_TYPE_ID]]
// LLVM:   br i1 %[[TYPE_ID_EQ]], label %[[BEGIN_CATCH:.*]], label %[[RESUME:.*]]
// LLVM: [[BEGIN_CATCH]]:
// LLVMCIR:   %[[EXN:.*]] = phi ptr [ %[[DISPATCH_EXN]], %[[DISPATCH]] ]
// LLVMCIR:   %{{.*}} = phi i32 [ %[[EH_SELECTOR]], %[[DISPATCH]] ]
// OGCG:   %[[EXN:.*]] = load ptr, ptr %[[EXN_SLOT]], align 8
// LLVM:   %[[EXN_PTR:.*]] = call ptr @__cxa_begin_catch(ptr %[[EXN]])
// LLVMCIR:   br label %[[CATCH_BODY:.*]]
// LLVMCIR: [[CATCH_BODY]]:
// LLVM:   %[[TMP_EXN:.*]] = load i32, ptr %[[EXN_PTR]], align 4
// LLVM:   store i32 %[[TMP_EXN]], ptr %[[E]], align 4
// LLVMCIR:   br label %[[END_CATCH:.*]]
// LLVMCIR: [[END_CATCH]]:
// LLVM:   call void @__cxa_end_catch()
// LLVMCIR:   br label %[[END_DISPATCH:.*]]
// LLVMCIR: [[END_DISPATCH]]:
// LLVMCIR:   br label %[[END_TRY:.*]]
// LLVMCIR: [[END_TRY]]:
// LLVM:   br label %[[TRY_CONT:.*]]
// OGCG: [[TRY_CONT]]:
// OGCG:   ret void
// LLVM: [[RESUME]]:
// LLVMCIR:   %[[RESUME_EXN:.*]] = phi ptr [ %[[DISPATCH_EXN]], %[[DISPATCH]] ]
// LLVMCIR:   %[[RESUME_SEL:.*]] = phi i32 [ %[[EH_SELECTOR]], %[[DISPATCH]] ]
// OGCG:   %[[RESUME_EXN:.*]] = load ptr, ptr %[[EXN_SLOT]], align 8
// OGCG:   %[[RESUME_SEL:.*]] = load i32, ptr %[[EH_SELECTOR_SLOT]], align 4
// LLVM:   %[[TMP_EXCEPTION_INFO:.*]] = insertvalue { ptr, i32 } poison, ptr %[[RESUME_EXN]], 0
// LLVM:   %[[EXCEPTION_INFO:.*]] = insertvalue { ptr, i32 } %[[TMP_EXCEPTION_INFO]], i32 %[[RESUME_SEL]], 1
// LLVM:   resume { ptr, i32 } %[[EXCEPTION_INFO]]
// LLVMCIR: [[TRY_CONT]]:
// LLVMCIR:   br label %[[DONE:.*]]
// LLVMCIR: [[DONE]]:
// LLVMCIR:   ret void
// OGCG: [[UNREACHABLE]]:
// OGCG:   unreachable

const void *init_catch_param_with_ref_to_nullptr() {
  try {
    division();
  } catch (const decltype(nullptr) &n) {
    return &n;
  }
  return nullptr;
}

// CIR: cir.func {{.*}} @_Z36init_catch_param_with_ref_to_nullptrv() {{.*}} personality(@__gxx_personality_v0)
// CIR:   %[[N_ADDR:.*]] = cir.alloca "n" {{.*}} const : !cir.ptr<!cir.ptr<!cir.ptr<!void>>>
// CIR:   } catch [type #cir.global_view<@_ZTIDn> : !cir.ptr<!u8i>] (%[[TOKEN:.*]]: !cir.eh_token {{.*}}) {
// CIR-NEXT: %[[CATCH_TOKEN:.*]], %[[EXN_PTR:.*]] = cir.begin_catch %[[TOKEN]] : !cir.eh_token -> (!cir.catch_token, !cir.ptr<!void>)
// CIR:     cir.init_catch_param reference %[[EXN_PTR]] to %[[N_ADDR]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!cir.ptr<!void>>>

// LLVM: define {{.*}} ptr @_Z36init_catch_param_with_ref_to_nullptrv() {{.*}} personality ptr @__gxx_personality_v0
// LLVM:   %[[EXN_PTR:.*]] = call ptr @__cxa_begin_catch(ptr %{{.*}})
// LLVM:   store ptr %[[EXN_PTR]], ptr %[[N_ADDR:.*]], align 8
// LLVM-NEXT:   %[[N:.*]] = load ptr, ptr %[[N_ADDR]], align 8
// LLVM-NEXT:   store ptr %[[N]], ptr %{{.*}}, align 8

const void *init_catch_param_with_ref_to_void_ptr() {
  try {
    division();
  } catch (void *const &p) {
    return &p;
  }
  return nullptr;
}

// CIR: cir.func {{.*}} @_Z37init_catch_param_with_ref_to_void_ptrv() {{.*}} personality(@__gxx_personality_v0)
// CIR:   %[[P_ADDR:.*]] = cir.alloca "p" {{.*}} const : !cir.ptr<!cir.ptr<!cir.ptr<!void>>>
// CIR:   } catch [type #cir.global_view<@_ZTIPv> : !cir.ptr<!u8i>] (%[[TOKEN:.*]]: !cir.eh_token {{.*}}) {
// CIR-NEXT: cir.construct_catch_param reference_to_pointer %[[TOKEN]] to %[[P_ADDR]] using : !cir.ptr<!cir.ptr<!cir.ptr<!void>>>
// CIR-NEXT: %[[CATCH_TOKEN:.*]], %[[EXN_PTR:.*]] = cir.begin_catch %[[TOKEN]] : !cir.eh_token -> (!cir.catch_token, !cir.ptr<!void>)
// CIR:     cir.init_catch_param reference_to_pointer %[[EXN_PTR]] to %[[P_ADDR]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!cir.ptr<!void>>>

// LLVM: define {{.*}} ptr @_Z37init_catch_param_with_ref_to_void_ptrv() {{.*}} personality ptr @__gxx_personality_v0
// LLVMCIR:   %[[EXN_OBJ:.*]] = getelementptr i8, ptr %[[EXN:.*]], i64 32
// LLVMCIR-NEXT:   store ptr %[[EXN_OBJ]], ptr %[[P_ADDR:.+]], align 8
// OGCG:   %[[EXN:.*]] = load ptr, ptr %{{.+}}, align 8
// LLVM-NEXT:   %{{.*}} = call ptr @__cxa_begin_catch(ptr %[[EXN]])
// LLVMCIR-NEXT:   br label %[[CATCH_BODY:.+]]
// LLVMCIR: [[CATCH_BODY]]:
// OGCG-NEXT:   %[[EXN_OBJ:.*]] = getelementptr i8, ptr %[[EXN]], i32 32
// OGCG-NEXT:   store ptr %[[EXN_OBJ]], ptr %[[P_ADDR:.+]], align 8
// LLVM-NEXT:   %{{.+}} = load ptr, ptr %[[P_ADDR]], align 8

const void *init_catch_param_with_ref_to_atomic_record_ptr() {
  try {
    division();
  } catch (_Atomic(Record) *const &p) {
    return &p;
  }
  return nullptr;
}

// CIR: cir.func {{.*}} @_Z46init_catch_param_with_ref_to_atomic_record_ptrv() {{.*}} personality(@__gxx_personality_v0)
// CIR:   %[[P_ADDR:.*]] = cir.alloca "p" {{.*}} const : !cir.ptr<!cir.ptr<!cir.ptr<!rec_Record>>>
// CIR:   } catch [type #cir.global_view<@_ZTIPU7_Atomic6Record> : !cir.ptr<!u8i>] (%[[TOKEN:.*]]: !cir.eh_token {{.*}}) {
// CIR-NEXT: cir.construct_catch_param reference_to_pointer %[[TOKEN]] to %[[P_ADDR]] using : !cir.ptr<!cir.ptr<!cir.ptr<!rec_Record>>>
// CIR-NEXT: %[[CATCH_TOKEN:.*]], %[[EXN_PTR:.*]] = cir.begin_catch %[[TOKEN]] : !cir.eh_token -> (!cir.catch_token, !cir.ptr<!void>)
// CIR:     cir.init_catch_param reference_to_pointer %[[EXN_PTR]] to %[[P_ADDR]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!cir.ptr<!rec_Record>>>

// LLVM: define {{.*}} ptr @_Z46init_catch_param_with_ref_to_atomic_record_ptrv() {{.*}} personality ptr @__gxx_personality_v0
// LLVMCIR:   %[[EXN_OBJ:.*]] = getelementptr i8, ptr %[[EXN:.*]], i64 32
// LLVMCIR-NEXT:   store ptr %[[EXN_OBJ]], ptr %[[P_ADDR:.+]], align 8
// OGCG:   %[[EXN:.*]] = load ptr, ptr %{{.+}}, align 8
// LLVM-NEXT:   %{{.*}} = call ptr @__cxa_begin_catch(ptr %[[EXN]])
// LLVMCIR-NEXT:   br label %[[CATCH_BODY:.+]]
// LLVMCIR: [[CATCH_BODY]]:
// OGCG-NEXT:   %[[EXN_OBJ:.*]] = getelementptr i8, ptr %[[EXN]], i32 32
// OGCG-NEXT:   store ptr %[[EXN_OBJ]], ptr %[[P_ADDR:.+]], align 8
// LLVM-NEXT:   %{{.+}} = load ptr, ptr %[[P_ADDR]], align 8

const void *init_catch_param_with_ref_to_ptr_to_member_function_ptr() {
  try {
    division();
  } catch (int (Record::**const &p)()) {
    return &p;
  }
  return nullptr;
}

// CIR: cir.func {{.*}} @_Z55init_catch_param_with_ref_to_ptr_to_member_function_ptrv() {{.*}} personality(@__gxx_personality_v0)
// CIR:   %[[P_ADDR:.*]] = cir.alloca "p" {{.*}} const : !cir.ptr<!cir.ptr<!cir.ptr<!rec_anon_struct>>>
// CIR:   } catch [type #cir.global_view<@_ZTIPM6RecordFivE> : !cir.ptr<!u8i>] (%[[TOKEN:.*]]: !cir.eh_token {{.*}}) {
// CIR-NEXT: cir.construct_catch_param reference_to_pointer %[[TOKEN]] to %[[P_ADDR]] using : !cir.ptr<!cir.ptr<!cir.ptr<!rec_anon_struct>>>
// CIR-NEXT: %[[CATCH_TOKEN:.*]], %[[EXN_PTR:.*]] = cir.begin_catch %[[TOKEN]] : !cir.eh_token -> (!cir.catch_token, !cir.ptr<!void>)
// CIR:     cir.init_catch_param reference_to_pointer %[[EXN_PTR]] to %[[P_ADDR]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!cir.ptr<!rec_anon_struct>>>

// LLVM: define {{.*}} ptr @_Z55init_catch_param_with_ref_to_ptr_to_member_function_ptrv() {{.*}} personality ptr @__gxx_personality_v0
// LLVMCIR:   %[[EXN_OBJ:.*]] = getelementptr i8, ptr %[[EXN:.*]], i64 32
// LLVMCIR-NEXT:   store ptr %[[EXN_OBJ]], ptr %[[P_ADDR:.+]], align 8
// OGCG:   %[[EXN:.*]] = load ptr, ptr %{{.+}}, align 8
// LLVM-NEXT:   %{{.*}} = call ptr @__cxa_begin_catch(ptr %[[EXN]])
// LLVMCIR-NEXT:   br label %[[CATCH_BODY:.+]]
// LLVMCIR: [[CATCH_BODY]]:
// OGCG-NEXT:   %[[EXN_OBJ:.*]] = getelementptr i8, ptr %[[EXN]], i32 32
// OGCG-NEXT:   store ptr %[[EXN_OBJ]], ptr %[[P_ADDR:.+]], align 8
// LLVM-NEXT:   %{{.+}} = load ptr, ptr %[[P_ADDR]], align 8

Record gr;

int init_catch_param_with_ref_to_member_function_ptr() {
  try {
    division();
  } catch (int (Record::*const &mfp)()) {
    return (gr.*mfp)();
  }
  return 0;
}

// CIR: cir.func {{.*}} @_Z48init_catch_param_with_ref_to_member_function_ptrv() {{.*}} personality(@__gxx_personality_v0)
// CIR:   %[[MFP_ADDR:.*]] = cir.alloca "mfp" {{.*}} const : !cir.ptr<!cir.ptr<!rec_anon_struct>>
// CIR:   } catch [type #cir.global_view<@_ZTIM6RecordFivE> : !cir.ptr<!u8i>] (%[[TOKEN:.*]]: !cir.eh_token {{.*}}) {
// CIR-NEXT: %[[CATCH_TOKEN:.*]], %[[EXN_PTR:.*]] = cir.begin_catch %[[TOKEN]] : !cir.eh_token -> (!cir.catch_token, !cir.ptr<!void>)
// CIR:     cir.init_catch_param reference %[[EXN_PTR]] to %[[MFP_ADDR]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!rec_anon_struct>>

// LLVM: define {{.*}} i32 @_Z48init_catch_param_with_ref_to_member_function_ptrv() {{.*}} personality ptr @__gxx_personality_v0
// LLVM:   %[[EXN_PTR:.*]] = call ptr @__cxa_begin_catch(ptr %{{.*}})
// LLVM:   store ptr %[[EXN_PTR]], ptr %[[MFP_ADDR:.*]], align 8
// LLVM-NEXT:   %[[MFP:.*]] = load ptr, ptr %[[MFP_ADDR]], align 8
// LLVM-NEXT:   %{{.*}} = load { i64, i64 }, ptr %[[MFP]], align 8

void init_catch_param_with_ref_to_ptr_to_ptr_to_record() {
  try {
    division();
  } catch (Record **&pp) {
  }
}

// CIR: cir.func {{.*}} @_Z49init_catch_param_with_ref_to_ptr_to_ptr_to_recordv(){{.*}} personality(@__gxx_personality_v0)
// CIR:   %[[PP_ADDR:.*]] = cir.alloca "pp" {{.*}} const : !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_Record>>>>
// CIR:   } catch [type #cir.global_view<@_ZTIPP6Record> : !cir.ptr<!u8i>] (%[[TOKEN:.*]]: !cir.eh_token {{.*}}) {
// CIR-NEXT: cir.construct_catch_param reference_to_pointer %[[TOKEN]] to %[[PP_ADDR]] using : !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_Record>>>>
// CIR-NEXT: %[[CATCH_TOKEN:.*]], %[[EXN_PTR:.*]] = cir.begin_catch %[[TOKEN]] : !cir.eh_token -> (!cir.catch_token, !cir.ptr<!void>)
// CIR:     cir.init_catch_param reference_to_pointer %[[EXN_PTR]] to %[[PP_ADDR]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!cir.ptr<!cir.ptr<!rec_Record>>>>

// LLVM: define {{.*}} void @_Z49init_catch_param_with_ref_to_ptr_to_ptr_to_recordv() {{.*}} personality ptr @__gxx_personality_v0
// OGCG:   %{{.*}} = alloca i32, align 4
// LLVM-NEXT:   %[[PP_ADDR:.*]] = alloca ptr, align 8
// LLVMCIR:   %[[EXN_OBJ:.*]] = getelementptr i8, ptr %[[EXN:.*]], i64 32
// LLVMCIR-NEXT:   store ptr %[[EXN_OBJ]], ptr %[[PP_ADDR]], align 8
// OGCG:   %[[EXN:.*]] = load ptr, ptr %{{.+}}, align 8
// LLVM-NEXT:   %{{.*}} = call ptr @__cxa_begin_catch(ptr %[[EXN]])
// LLVMCIR-NEXT:   br label %[[CATCH_BODY:.+]]
// LLVMCIR: [[CATCH_BODY]]:
// LLVMCIR-NEXT:   br label %[[END_CATCH:.*]]
// LLVMCIR: [[END_CATCH]]:
// LLVMCIR-NEXT:   call void @__cxa_end_catch()
// OGCG-NEXT:   %[[EXN_OBJ:.*]] = getelementptr i8, ptr %[[EXN]], i32 32
// OGCG-NEXT:   store ptr %[[EXN_OBJ]], ptr %[[PP_ADDR]], align 8

union Union {
  int i;
};

void init_catch_param_with_ref_to_union_ptr() {
  try {
    division();
  } catch (Union *&up) {
  }
}

// CIR: cir.func {{.*}} @_Z38init_catch_param_with_ref_to_union_ptrv(){{.*}} personality(@__gxx_personality_v0)
// CIR:   %[[UP_ADDR:.*]] = cir.alloca "up" {{.*}} const : !cir.ptr<!cir.ptr<!cir.ptr<!rec_Union>>>
// CIR:   } catch [type #cir.global_view<@_ZTIP5Union> : !cir.ptr<!u8i>] (%[[TOKEN:.*]]: !cir.eh_token {{.*}}) {
// CIR-NEXT: %[[CATCH_TOKEN:.*]], %[[EXN_PTR:.*]] = cir.begin_catch %[[TOKEN]] : !cir.eh_token -> (!cir.catch_token, !cir.ptr<!void>)
// CIR:     cir.init_catch_param reference_to_record_pointer %[[EXN_PTR]] to %[[UP_ADDR]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!cir.ptr<!rec_Union>>>

// LLVM: define {{.*}} void @_Z38init_catch_param_with_ref_to_union_ptrv() {{.*}} personality ptr @__gxx_personality_v0
// LLVMCIR:   %[[TMP:.*]] = alloca ptr, align 8
// LLVMCIR-NEXT:   %[[UP_ADDR:.*]] = alloca ptr, align 8
// OGCG:   %{{.*}} = alloca i32, align 4
// OGCG-NEXT:   %[[UP_ADDR:.*]] = alloca ptr, align 8
// OGCG-NEXT:   %[[TMP:.*]] = alloca ptr, align 8
// LLVM:   %[[EXN_PTR:.*]] = call ptr @__cxa_begin_catch(ptr %{{.*}})
// LLVM:   store ptr %[[EXN_PTR]], ptr %[[TMP]], align 8
// LLVM-NEXT:   store ptr %[[TMP]], ptr %[[UP_ADDR]], align 8
