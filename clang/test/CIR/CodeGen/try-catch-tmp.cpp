// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

int division();

void calling_division_inside_try_block() {
  try {
    division();
  } catch (...) {
  }
}

// CIR: cir.scope {
// CIR:   cir.try {
// CIR:       %[[CALL:.*]] = cir.call @_Z8divisionv() : () -> !s32i
// CIR:       cir.yield
// CIR:   } catch all {
// CIR:       cir.yield
// CIR:   }
// CIR: }

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
