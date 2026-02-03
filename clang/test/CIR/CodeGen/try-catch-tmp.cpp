// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

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
// CIR:       %[[CALL:.*]] = cir.call @_Z8divisionv() : () -> !s32i
// CIR:       cir.yield
// CIR:   } catch all {
// CIR:       %[[CATCH_PARAM:.*]] = cir.catch_param : !cir.ptr<!void>
// CIR:       cir.yield
// CIR:   }
// CIR: }


// OGCG: define {{.*}} void @_Z34call_function_inside_try_catch_allv() #0 personality ptr @__gxx_personality_v0
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
// CIR:     %[[CALL:.*]] = cir.call @_Z8divisionv() : () -> !s32i
// CIR:     cir.yield
// CIR:   } catch [type #cir.global_view<@_ZTIi> : !cir.ptr<!u8i>] {
// CIR:     %[[CATCH_PARAM:.*]] = cir.catch_param : !cir.ptr<!s32i>
// CIR      %[[TMP_CATCH_PARAM:.*]] = cir.load {{.*}} %[[CATCH_PRAM]] : !cir.ptr<!s32i>, !s32i
// CIR      cir.store {{.*}} %[[TMP_CATCH_PARAM]], %[[EXCEPTION_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:     cir.yield
// CIR:   } unwind {
// CIR:     cir.resume
// CIR:   }
// CIR: }

// OGCG: define {{.*}} void @_Z50call_function_inside_try_catch_with_exception_typev() #0 personality ptr @__gxx_personality_v0
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
// CIR:     %[[CALL:.*]] = cir.call @_Z8divisionv() : () -> !s32i
// CIR:     cir.yield
// CIR:   } catch [type #cir.global_view<@_ZTICi> : !cir.ptr<!u8i>] {
// CIR:     %[[CATCH_PARAM:.*]] = cir.catch_param : !cir.ptr<!cir.complex<!s32i>>
// CIR:     %[[TMP_CATCH_PARAM:.*]] = cir.load {{.*}} %[[CATCH_PARAM]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR:     cir.store {{.*}} %[[TMP_CATCH_PARAM]], %[[EXCEPTION_ADDR]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>
// CIR:     cir.yield
// CIR:   } unwind {
// CIR:     cir.resume
// CIR:   }
// CIR: }

// OGCG: define {{.*}} void @_Z58call_function_inside_try_catch_with_complex_exception_typev() #0 personality ptr @__gxx_personality_v0
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
// CIR:     %[[CALL:.*]] = cir.call @_Z8divisionv() : () -> !s32i
// CIR:     cir.yield
// CIR:   } catch [type #cir.global_view<@_ZTIPi> : !cir.ptr<!u8i>] {
// CIR:     %[[CATCH_PARAM:.*]] = cir.catch_param : !cir.ptr<!s32i>
// CIR:     cir.store {{.*}} %[[CATCH_PARAM]], %[[E_ADDR]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR:     cir.yield
// CIR:   } unwind {
// CIR:     cir.resume
// CIR:   }
// CIR: }

// OGCG: define {{.*}} void @_Z56call_function_inside_try_catch_with_array_exception_typev() #0 personality ptr @__gxx_personality_v0
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
// CIR:     %[[CALL:.*]] = cir.call @_Z8divisionv() : () -> !s32i
// CIR:     cir.yield
// CIR:   } catch [type #cir.global_view<@_ZTIi> : !cir.ptr<!u8i>] {
// CIR:     %[[CATCH_PARAM:.*]] = cir.catch_param : !cir.ptr<!s32i>
// CIR      %[[TMP_CATCH_PARAM:.*]] = cir.load {{.*}} %[[CATCH_PRAM]] : !cir.ptr<!s32i>, !s32i
// CIR      cir.store {{.*}} %[[TMP_CATCH_PARAM]], %[[EXCEPTION_ADDR]] : !s32i, !cir.ptr<!s32i>
// CIR:     cir.yield
// CIR:   } catch all {
// CIR:     %[[CATCH_PARAM]] = cir.catch_param : !cir.ptr<!void>
// CIR:     cir.yield
// CIR:   }
// CIR: }

// OGCG: define {{.*}} void @_Z64call_function_inside_try_catch_with_exception_type_and_catch_allv() #0 personality ptr @__gxx_personality_v0
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
