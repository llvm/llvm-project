// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

int if0(bool a) {

  if (a)
    return 2;

  return 3;

}

// CIR: cir.func @_Z3if0b(%arg0: !cir.bool loc({{.*}})) -> !s32i
// CIR: cir.scope {
// CIR:   %4 = cir.load{{.*}} %0 : !cir.ptr<!cir.bool>, !cir.bool
// CIR-NEXT: cir.if %4 {
// CIR-NEXT:   %5 = cir.const #cir.int<2> : !s32i
// CIR-NEXT:   cir.store{{.*}} %5, %1 : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:   %6 = cir.load{{.*}} %1 : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:   cir.return %6 : !s32i
// CIR-NEXT:   }
// CIR-NEXT:  }


// LLVM: define i32 @_Z3if0b(i1 %0)
// LLVM:   br label %[[ENTRY:.*]]
// LLVM: [[ENTRY]]:
// LLVM:   %6 = load i8, ptr %2, align 1
// LLVM:   %7 = trunc i8 %6 to i1
// LLVM:   br i1 %7, label %[[THEN:.*]], label %[[END:.*]]
// LLVM: [[THEN]]:
// LLVM:   store i32 2, ptr %3, align 4
// LLVM:   %9 = load i32, ptr %3, align 4
// LLVM:   ret i32 %9
// LLVM: [[END]]:
// LLVM:   br label %[[LABEL4:.*]]
// LLVM: [[LABEL4]]:
// LLVM:   store i32 3, ptr %3, align 4
// LLVM:   %12 = load i32, ptr %3, align 4
// LLVM:   ret i32 %12

// OGCG: define dso_local noundef i32 @_Z3if0b(i1 noundef zeroext %a)
// OGCG: entry:
// OGCG:   %[[RETVAL:.*]] = alloca i32, align 4
// OGCG:   %[[A_ADDR:.*]] = alloca i8, align 1
// OGCG:   %[[STOREDV:.*]] = zext i1 %a to i8
// OGCG:   store i8 %[[STOREDV]], ptr %[[A_ADDR]], align 1
// OGCG:   %[[LOADTMP:.*]] = load i8, ptr %[[A_ADDR]], align 1
// OGCG:   %[[LOADEDV:.*]] = trunc i8 %[[LOADTMP]] to i1
// OGCG:   br i1 %[[LOADEDV]], label %[[THEN_LABEL:.*]], label %[[END_LABEL:.*]]
// OGCG: [[THEN_LABEL]]:
// OGCG:   store i32 2, ptr %[[RETVAL]], align 4
// OGCG:   br label %[[RETURN_LABEL:.*]]
// OGCG: [[END_LABEL]]:
// OGCG:   store i32 3, ptr %[[RETVAL]], align 4
// OGCG:   br label %[[RETURN_LABEL]]
// OGCG: [[RETURN_LABEL]]:
// OGCG:   %[[FINALLOAD:.*]] = load i32, ptr %[[RETVAL]], align 4
// OGCG:   ret i32 %[[FINALLOAD]]

void if1(int a) {
  int x = 0;
  if (a) {
    x = 3;
  } else {
    x = 4;
  }
}

// CIR: cir.func @_Z3if1i(%arg0: !s32i loc({{.*}}))
// CIR: cir.scope {
// CIR:   %3 = cir.load{{.*}} %0 : !cir.ptr<!s32i>, !s32i
// CIR:   %4 = cir.cast(int_to_bool, %3 : !s32i), !cir.bool
// CIR-NEXT:   cir.if %4 {
// CIR-NEXT:     %5 = cir.const #cir.int<3> : !s32i
// CIR-NEXT:     cir.store{{.*}} %5, %1 : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:   } else {
// CIR-NEXT:     %5 = cir.const #cir.int<4> : !s32i
// CIR-NEXT:     cir.store{{.*}} %5, %1 : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:   }
// CIR: }

// LLVM: define void @_Z3if1i(i32 %0)
// LLVM: %[[A:.*]] = alloca i32, i64 1, align 4
// LLVM: %[[X:.*]] = alloca i32, i64 1, align 4
// LLVM: store i32 %0, ptr %[[A]], align 4
// LLVM: store i32 0, ptr %[[X]], align 4
// LLVM: br label %[[ENTRY:.*]]
// LLVM: [[ENTRY]]:
// LLVM:   %[[LOADED:.*]] = load i32, ptr %[[A]], align 4
// LLVM:   %[[COND:.*]] = icmp ne i32 %[[LOADED]], 0
// LLVM:   br i1 %[[COND]], label %[[THEN:.*]], label %[[ELSE:.*]]
// LLVM: [[THEN]]:
// LLVM:   store i32 3, ptr %[[X]], align 4
// LLVM:   br label %[[END:.*]]
// LLVM: [[ELSE]]:
// LLVM:   store i32 4, ptr %[[X]], align 4
// LLVM:   br label %[[END]]
// LLVM: [[END]]:
// LLVM:   br label %[[EXIT:.*]]
// LLVM: [[EXIT]]:
// LLVM:   ret void

// OGCG: define dso_local void @_Z3if1i(i32 noundef %[[A:.*]])
// OGCG: entry:
// OGCG:   %[[A_ADDR:.*]] = alloca i32, align 4
// OGCG:   %[[X:.*]] = alloca i32, align 4
// OGCG:   store i32 %[[A]], ptr %[[A_ADDR]], align 4
// OGCG:   store i32 0, ptr %[[X]], align 4
// OGCG:   %[[LOADED_A:.*]] = load i32, ptr %[[A_ADDR]], align 4
// OGCG:   %[[TOBOOL:.*]] = icmp ne i32 %[[LOADED_A]], 0
// OGCG:   br i1 %[[TOBOOL]], label %[[THEN_LABEL:.*]], label %[[ELSE_LABEL:.*]]
// OGCG: [[THEN_LABEL]]:
// OGCG:   store i32 3, ptr %[[X]], align 4
// OGCG:   br label %[[END_LABEL:.*]]
// OGCG: [[ELSE_LABEL]]:
// OGCG:   store i32 4, ptr %[[X]], align 4
// OGCG:   br label %[[END_LABEL]]
// OGCG: [[END_LABEL]]:
// OGCG:   ret void

void if2(int a, bool b, bool c) {
  int x = 0;
  if (a) {
    x = 3;
    if (b) {
      x = 8;
    }
  } else {
    if (c) {
      x = 14;
    }
    x = 4;
  }
}

// CIR: cir.func @_Z3if2ibb(%arg0: !s32i loc({{.*}}), %arg1: !cir.bool loc({{.*}}), %arg2: !cir.bool loc({{.*}}))
// CIR: cir.scope {
// CIR:   %5 = cir.load{{.*}} %0 : !cir.ptr<!s32i>, !s32i
// CIR:   %6 = cir.cast(int_to_bool, %5 : !s32i), !cir.bool
// CIR:   cir.if %6 {
// CIR:     %7 = cir.const #cir.int<3> : !s32i
// CIR:     cir.store{{.*}} %7, %3 : !s32i, !cir.ptr<!s32i>
// CIR:     cir.scope {
// CIR:       %8 = cir.load{{.*}} %1 : !cir.ptr<!cir.bool>, !cir.bool
// CIR-NEXT:       cir.if %8 {
// CIR-NEXT:         %9 = cir.const #cir.int<8> : !s32i
// CIR-NEXT:         cir.store{{.*}} %9, %3 : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:       }
// CIR:     }
// CIR:   } else {
// CIR:     cir.scope {
// CIR:       %8 = cir.load{{.*}} %2 : !cir.ptr<!cir.bool>, !cir.bool
// CIR-NEXT:       cir.if %8 {
// CIR-NEXT:         %9 = cir.const #cir.int<14> : !s32i
// CIR-NEXT:         cir.store{{.*}} %9, %3 : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:       }
// CIR:     }
// CIR:     %7 = cir.const #cir.int<4> : !s32i
// CIR:     cir.store{{.*}} %7, %3 : !s32i, !cir.ptr<!s32i>
// CIR:   }
// CIR: }

// LLVM: define void @_Z3if2ibb(i32 %[[A:.*]], i1 %[[B:.*]], i1 %[[C:.*]])
// LLVM:   %[[VARA:.*]] = alloca i32, i64 1, align 4
// LLVM:   %[[VARB:.*]] = alloca i8, i64 1, align 1
// LLVM:   %[[VARC:.*]] = alloca i8, i64 1, align 1
// LLVM:   %[[VARX:.*]] = alloca i32, i64 1, align 4
// LLVM:   store i32 %[[A]], ptr %[[VARA]], align 4
// LLVM:   %[[B_EXT:.*]] = zext i1 %[[B]] to i8
// LLVM:   store i8 %[[B_EXT]], ptr %[[VARB]], align 1
// LLVM:   %[[C_EXT:.*]] = zext i1 %[[C]] to i8
// LLVM:   store i8 %[[C_EXT]], ptr %[[VARC]], align 1
// LLVM:   store i32 0, ptr %[[VARX]], align 4
// LLVM:   br label %[[ENTRY:.*]]
// LLVM: [[ENTRY]]:
// LLVM:   %[[LOAD_A:.*]] = load i32, ptr %[[VARA]], align 4
// LLVM:   %[[CMP_A:.*]] = icmp ne i32 %[[LOAD_A]], 0
// LLVM:   br i1 %[[CMP_A]], label %[[IF_THEN:.*]], label %[[IF_ELSE:.*]]
// LLVM: [[IF_THEN]]:
// LLVM:   store i32 3, ptr %[[VARX]], align 4
// LLVM:   br label %[[LABEL14:.*]]
// LLVM: [[LABEL14]]:
// LLVM:   %[[LOAD_B:.*]] = load i8, ptr %[[VARB]], align 1
// LLVM:   %[[TRUNC_B:.*]] = trunc i8 %[[LOAD_B]] to i1
// LLVM:   br i1 %[[TRUNC_B]], label %[[IF_THEN2:.*]], label %[[IF_END2:.*]]
// LLVM: [[IF_THEN2]]:
// LLVM:   store i32 8, ptr %[[VARX]], align 4
// LLVM:   br label %[[IF_END2]]
// LLVM: [[IF_END2]]:
// LLVM:   br label %[[LABEL19:.*]]
// LLVM: [[LABEL19]]:
// LLVM:   br label %[[LABEL27:.*]]
// LLVM: [[IF_ELSE]]:
// LLVM:   br label %[[LABEL21:.*]]
// LLVM: [[LABEL21]]:
// LLVM:   %[[LOAD_C:.*]] = load i8, ptr %[[VARC]], align 1
// LLVM:   %[[TRUNC_C:.*]] = trunc i8 %[[LOAD_C]] to i1
// LLVM:   br i1 %[[TRUNC_C]], label %[[IF_THEN3:.*]], label %[[IF_END3:.*]]
// LLVM: [[IF_THEN3]]:
// LLVM:   store i32 14, ptr %[[VARX]], align 4
// LLVM:   br label %[[IF_END3]]
// LLVM: [[IF_END3]]:
// LLVM:   br label %[[LABEL26:.*]]
// LLVM: [[LABEL26]]:
// LLVM:   store i32 4, ptr %[[VARX]], align 4
// LLVM:   br label %[[LABEL27]]
// LLVM: [[LABEL27]]:
// LLVM:   br label %[[LABEL28:.*]]
// LLVM: [[LABEL28]]:
// LLVM:   ret void

// OGCG: define dso_local void @_Z3if2ibb(i32 noundef %[[A:.*]], i1 noundef zeroext %[[B:.*]], i1 noundef zeroext %[[C:.*]])
// OGCG: entry:
// OGCG:   %[[A_ADDR:.*]] = alloca i32, align 4
// OGCG:   %[[B_ADDR:.*]] = alloca i8, align 1
// OGCG:   %[[C_ADDR:.*]] = alloca i8, align 1
// OGCG:   %[[X:.*]] = alloca i32, align 4
// OGCG:   store i32 %[[A]], ptr %[[A_ADDR]], align 4
// OGCG:   %[[B_EXT:.*]] = zext i1 %[[B]] to i8
// OGCG:   store i8 %[[B_EXT]], ptr %[[B_ADDR]], align 1
// OGCG:   %[[C_EXT:.*]] = zext i1 %[[C]] to i8
// OGCG:   store i8 %[[C_EXT]], ptr %[[C_ADDR]], align 1
// OGCG:   store i32 0, ptr %[[X]], align 4
// OGCG:   %[[A_VAL:.*]] = load i32, ptr %[[A_ADDR]], align 4
// OGCG:   %[[A_BOOL:.*]] = icmp ne i32 %[[A_VAL]], 0
// OGCG:   br i1 %[[A_BOOL]], label %[[IF_THEN:.*]], label %[[IF_ELSE:.*]]
// OGCG: [[IF_THEN]]:
// OGCG:   store i32 3, ptr %[[X]], align 4
// OGCG:   %[[B_LOAD:.*]] = load i8, ptr %[[B_ADDR]], align 1
// OGCG:   %[[B_TRUNC:.*]] = trunc i8 %[[B_LOAD]] to i1
// OGCG:   br i1 %[[B_TRUNC]], label %[[IF_THEN2:.*]], label %[[IF_END:.*]]
// OGCG: [[IF_THEN2]]:
// OGCG:   store i32 8, ptr %[[X]], align 4
// OGCG:   br label %[[IF_END]]
// OGCG: [[IF_END]]:
// OGCG:   br label %[[IF_END6:.*]]
// OGCG: [[IF_ELSE]]:
// OGCG:   %[[C_LOAD:.*]] = load i8, ptr %[[C_ADDR]], align 1
// OGCG:   %[[C_TRUNC:.*]] = trunc i8 %[[C_LOAD]] to i1
// OGCG:   br i1 %[[C_TRUNC]], label %[[IF_THEN4:.*]], label %[[IF_END5:.*]]
// OGCG: [[IF_THEN4]]:
// OGCG:   store i32 14, ptr %[[X]], align 4
// OGCG:   br label %[[IF_END5]]
// OGCG: [[IF_END5]]:
// OGCG:   store i32 4, ptr %[[X]], align 4
// OGCG:   br label %[[IF_END6]]
// OGCG: [[IF_END6]]:
// OGCG:   ret void

int if_init() {
  if (int x = 42 ; x) {
    return x + 1; // x should be visible here
  } else {
    return x - 1; // x should also be visible here
  }
}

// CIR: cir.func @_Z7if_initv() -> !s32i
// CIR: %[[RETVAL:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>
// CIR: cir.scope {
// CIR:   %[[X:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>,
// CIR:   %[[CONST42:.*]] = cir.const #cir.int<42> : !s32i
// CIR:   cir.store{{.*}} %[[CONST42]], %[[X]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[X_VAL:.*]] = cir.load{{.*}} %[[X]] : !cir.ptr<!s32i>, !s32i
// CIR:   %[[COND:.*]] = cir.cast(int_to_bool, %[[X_VAL]] : !s32i), !cir.bool
// CIR:   cir.if %[[COND]] {
// CIR:     %[[X_IF:.*]] = cir.load{{.*}} %[[X]] : !cir.ptr<!s32i>, !s32i
// CIR:     %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR:     %[[ADD:.*]] = cir.binop(add, %[[X_IF]], %[[ONE]]) nsw : !s32i
// CIR:     cir.store{{.*}} %[[ADD]], %[[RETVAL]] : !s32i, !cir.ptr<!s32i>
// CIR:     %[[RETVAL_LOAD1:.*]] = cir.load{{.*}} %[[RETVAL]] : !cir.ptr<!s32i>, !s32i
// CIR:     cir.return %[[RETVAL_LOAD1]] : !s32i
// CIR:   } else {
// CIR:     %[[X_ELSE:.*]] = cir.load{{.*}} %[[X]] : !cir.ptr<!s32i>, !s32i
// CIR:     %[[ONE2:.*]] = cir.const #cir.int<1> : !s32i
// CIR:     %[[SUB:.*]] = cir.binop(sub, %[[X_ELSE]], %[[ONE2]]) nsw : !s32i
// CIR:     cir.store{{.*}} %[[SUB]], %[[RETVAL]] : !s32i, !cir.ptr<!s32i>
// CIR:     %[[RETVAL_LOAD2:.*]] = cir.load{{.*}} %[[RETVAL]] : !cir.ptr<!s32i>, !s32i
// CIR:     cir.return %[[RETVAL_LOAD2]] : !s32i
// CIR:   }
// CIR: }

// LLVM: define i32 @_Z7if_initv()
// LLVM: %[[X:.*]] = alloca i32, i64 1, align 4
// LLVM: %[[RETVAL:.*]] = alloca i32, i64 1, align 4
// LLVM: store i32 42, ptr %[[X]], align 4
// LLVM: %[[X_VAL:.*]] = load i32, ptr %[[X]], align 4
// LLVM: %[[COND:.*]] = icmp ne i32 %[[X_VAL]], 0
// LLVM: br i1 %[[COND]], label %[[THEN:.*]], label %[[ELSE:.*]]
// LLVM: [[THEN]]:
// LLVM:   %[[X_LOAD:.*]] = load i32, ptr %[[X]], align 4
// LLVM:   %[[ADD:.*]] = add nsw i32 %[[X_LOAD]], 1
// LLVM:   store i32 %[[ADD]], ptr %[[RETVAL]], align 4
// LLVM:   %[[RETVAL_LOAD1:.*]] = load i32, ptr %[[RETVAL]], align 4
// LLVM:   ret i32 %[[RETVAL_LOAD1]]
// LLVM: [[ELSE]]:
// LLVM:   %[[X_LOAD2:.*]] = load i32, ptr %[[X]], align 4
// LLVM:   %[[SUB:.*]] = sub nsw i32 %[[X_LOAD2]], 1
// LLVM:   store i32 %[[SUB]], ptr %[[RETVAL]], align 4
// LLVM:   %[[RETVAL_LOAD2:.*]] = load i32, ptr %[[RETVAL]], align 4
// LLVM:   ret i32 %[[RETVAL_LOAD2]]

// OGCG: define dso_local noundef i32 @_Z7if_initv()
// OGCG: entry:
// OGCG:   %[[RETVAL:.*]] = alloca i32, align 4
// OGCG:   %[[X:.*]] = alloca i32, align 4
// OGCG:   store i32 42, ptr %[[X]], align 4
// OGCG:   %[[X_VAL:.*]] = load i32, ptr %[[X]], align 4
// OGCG:   %[[COND:.*]] = icmp ne i32 %[[X_VAL]], 0
// OGCG:   br i1 %[[COND]], label %[[THEN:.*]], label %[[ELSE:.*]]
// OGCG: [[THEN]]:
// OGCG:   %[[X_LOAD:.*]] = load i32, ptr %[[X]], align 4
// OGCG:   %[[ADD:.*]] = add nsw i32 %[[X_LOAD]], 1
// OGCG:   store i32 %[[ADD]], ptr %[[RETVAL]], align 4
// OGCG:   br label %[[RETURN:.*]]
// OGCG: [[ELSE]]:
// OGCG:   %[[X_LOAD2:.*]] = load i32, ptr %[[X]], align 4
// OGCG:   %[[SUB:.*]] = sub nsw i32 %[[X_LOAD2]], 1
// OGCG:   store i32 %[[SUB]], ptr %[[RETVAL]], align 4
// OGCG:   br label %[[RETURN]]
// OGCG: [[RETURN]]:
// OGCG:   %[[RETVAL_FINAL:.*]] = load i32, ptr %[[RETVAL]], align 4
// OGCG:   ret i32 %[[RETVAL_FINAL]]
