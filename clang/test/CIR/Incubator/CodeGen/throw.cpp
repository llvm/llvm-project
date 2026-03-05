// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -mconstructor-aliases -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir --check-prefix=CIR %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -mconstructor-aliases -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll --check-prefix=LLVM %s

double d(int a, int b) {
   if (b == 0)
      throw "Division by zero condition!";
   return (a/b);
}

//      CIR: cir.if
// CIR-NEXT:   %[[ADDR:.*]] = cir.alloc.exception 8
// CIR-NEXT:   %[[STR:.*]] = cir.get_global @".str" : !cir.ptr<!cir.array<!s8i x 28>>
// CIR-NEXT:   %[[STR_ADD:.*]] = cir.cast array_to_ptrdecay %[[STR]] : !cir.ptr<!cir.array<!s8i x 28>> -> !cir.ptr<!s8i>
// CIR-NEXT:   cir.store{{.*}} %[[STR_ADD]], %[[ADDR]] : !cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>>
// CIR-NEXT:   cir.throw %[[ADDR]] : !cir.ptr<!cir.ptr<!s8i>>, @_ZTIPKc
// CIR-NEXT:   cir.unreachable
// CIR-NEXT: ^bb1:  // no predecessors
// CIR-NEXT:   cir.yield
// CIR-NEXT: }

// LLVM: %[[ADDR:.*]] = call ptr @__cxa_allocate_exception(i64 8)
// LLVM: store ptr @.str, ptr %[[ADDR]], align 16
// LLVM: call void @__cxa_throw(ptr %[[ADDR]], ptr @_ZTIPKc, ptr null)
// LLVM: unreachable

struct S {
  S() {}
};

void refoo1() {
  int r = 1;
  try {
    S s;
    throw;
  } catch (...) {
    ++r;
  }
}

// CIR-LABEL: @_Z6refoo1v()
// CIR:   %[[V0:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["r", init] {alignment = 4 : i64}
// CIR:   %[[V1:.*]] = cir.const #cir.int<1> : !s32i
// CIR:   cir.store{{.*}} %[[V1]], %[[V0]] : !s32i, !cir.ptr<!s32i>
// CIR:   cir.scope {
// CIR:     %[[V2:.*]] = cir.alloca !rec_S, !cir.ptr<!rec_S>, ["s", init] {alignment = 1 : i64}
// CIR:     cir.try {
// CIR:       cir.call exception @_ZN1SC2Ev(%[[V2]]) : (!cir.ptr<!rec_S>) -> ()
// CIR:       cir.call exception @__cxa_rethrow() : () -> ()
// CIR:       cir.unreachable
// CIR:     ^bb1:  // no predecessors
// CIR:       cir.yield
// CIR:     } catch [type #cir.all {
// CIR:       %[[V3:.*]] = cir.catch_param -> !cir.ptr<!void>
// CIR:       %[[V4:.*]] = cir.load{{.*}} %[[V0]] : !cir.ptr<!s32i>, !s32i
// CIR:       %[[V5:.*]] = cir.unary(inc, %[[V4]]) nsw : !s32i, !s32i
// CIR:       cir.store{{.*}} %[[V5]], %[[V0]] : !s32i, !cir.ptr<!s32i>
// CIR:       cir.yield
// CIR:     }]
// CIR:   }
// CIR:   cir.return
// CIR: }

// LLVM: define dso_local void @_Z6refoo1v()
// LLVM:   %[[V1:.*]] = alloca %struct.S, i64 1, align 1
// LLVM:   %[[V2:.*]] = alloca i32, i64 1, align 4
// LLVM:   store i32 1, ptr %[[V2]], align 4
// LLVM:   br label %[[B3:.*]]
// LLVM: [[B3]]:
// LLVM:   br label %[[B4:.*]]
// LLVM: [[B4]]:
// LLVM:   invoke void @_ZN1SC2Ev(ptr %[[V1]])
// LLVM:           to label %[[B5:.*]] unwind label %[[B7:.*]]
// LLVM: [[B5]]:
// LLVM:   invoke void @__cxa_rethrow()
// LLVM:           to label %[[B6:.*]] unwind label %[[B11:.*]]
// LLVM: [[B6]]:
// LLVM:   unreachable
// LLVM: [[B7]]:
// LLVM:   %[[V8:.*]] = landingpad { ptr, i32 }
// LLVM:           catch ptr null
// LLVM:   %[[V9:.*]] = extractvalue { ptr, i32 } %[[V8]], 0
// LLVM:   %[[V10:.*]] = extractvalue { ptr, i32 } %[[V8]], 1
// LLVM:   br label %[[B15:.*]]
// LLVM: [[B11]]:
// LLVM:   %[[V12:.*]] = landingpad { ptr, i32 }
// LLVM:           catch ptr null
// LLVM:   %[[V13:.*]] = extractvalue { ptr, i32 } %[[V12]], 0
// LLVM:   %[[V14:.*]] = extractvalue { ptr, i32 } %[[V12]], 1
// LLVM:   br label %[[B15:.*]]
// LLVM: [[B15]]:
// LLVM:   %[[V16:.*]] = phi ptr [ %[[V9]], %[[B7]] ], [ %[[V13]], %[[B11]] ]
// LLVM:   %[[V17:.*]] = call ptr @__cxa_begin_catch(ptr %[[V16]])
// LLVM:   %[[V18:.*]] = load i32, ptr %[[V2]], align 4
// LLVM:   %[[V19:.*]] = add nsw i32 %[[V18]], 1
// LLVM:   store i32 %[[V19]], ptr %[[V2]], align 4
// LLVM:   call void @__cxa_end_catch()

void refoo2() {
  int r = 1;
  try {
    for (int i = 0; i < 5; i++) {
      S s;
      throw;
    }
    S s;
  } catch (...) {
    ++r;
  }
}

// CIR-LABEL: @_Z6refoo2v()
// CIR:   %[[V0:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["r", init] {alignment = 4 : i64}
// CIR:   %[[V1:.*]] = cir.const #cir.int<1> : !s32i
// CIR:   cir.store{{.*}} %[[V1]], %[[V0]] : !s32i, !cir.ptr<!s32i>
// CIR:   cir.scope {
// CIR:     %[[V2:.*]] = cir.alloca !rec_S, !cir.ptr<!rec_S>, ["s", init] {alignment = 1 : i64}
// CIR:     cir.try {
// CIR:       cir.scope {
// CIR:         %[[V3:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init] {alignment = 4 : i64}
// CIR:         %[[V4:.*]] = cir.const #cir.int<0> : !s32i
// CIR:         cir.store{{.*}} %[[V4]], %[[V3]] : !s32i, !cir.ptr<!s32i>
// CIR:         cir.for : cond {
// CIR:           %[[V5:.*]] = cir.load{{.*}} %[[V3]] : !cir.ptr<!s32i>, !s32i
// CIR:           %[[V6:.*]] = cir.const #cir.int<5> : !s32i
// CIR:           %[[V7:.*]] = cir.cmp(lt, %[[V5]], %[[V6]]) : !s32i, !cir.bool
// CIR:           cir.condition(%[[V7]])
// CIR:         } body {
// CIR:           cir.scope {
// CIR:             %[[V5:.*]] = cir.alloca !rec_S, !cir.ptr<!rec_S>, ["s", init] {alignment = 1 : i64}
// CIR:             cir.call exception @_ZN1SC2Ev(%[[V5]]) : (!cir.ptr<!rec_S>) -> ()
// CIR:             cir.call exception @__cxa_rethrow() : () -> ()
// CIR:             cir.unreachable
// CIR:           ^bb1:  // no predecessors
// CIR:             cir.yield
// CIR:           }
// CIR:           cir.yield
// CIR:         } step {
// CIR:           %[[V5:.*]] = cir.load{{.*}} %[[V3]] : !cir.ptr<!s32i>, !s32i
// CIR:           %[[V6:.*]] = cir.unary(inc, %[[V5]]) nsw : !s32i, !s32i
// CIR:           cir.store{{.*}} %[[V6]], %[[V3]] : !s32i, !cir.ptr<!s32i>
// CIR:           cir.yield
// CIR:         }
// CIR:       }
// CIR:       cir.call exception @_ZN1SC2Ev(%[[V2]]) : (!cir.ptr<!rec_S>) -> ()
// CIR:       cir.yield
// CIR:     } catch [type #cir.all {
// CIR:       %[[V3:.*]] = cir.catch_param -> !cir.ptr<!void>
// CIR:       %[[V4:.*]] = cir.load{{.*}} %[[V0]] : !cir.ptr<!s32i>, !s32i
// CIR:       %[[V5:.*]] = cir.unary(inc, %[[V4]]) nsw : !s32i, !s32i
// CIR:       cir.store{{.*}} %[[V5]], %[[V0]] : !s32i, !cir.ptr<!s32i>
// CIR:       cir.yield
// CIR:     }]
// CIR:   }
// CIR:   cir.return
// CIR: }

// LLVM: {{.*}}:
// LLVM:   invoke void @_ZN1SC2Ev(ptr %[[V3:.*]])
// LLVM:           to label %[[B13:.*]] unwind label %[[B22:.*]]
// LLVM: [[B13]]:
// LLVM:   invoke void @__cxa_rethrow()
// LLVM:           to label %[[B14:.*]] unwind label %[[B26:.*]]
// LLVM: [[B14]]:
// LLVM:   unreachable
// LLVM: [[B15]]:
// LLVM:   br label %[[B16:.*]]
// LLVM: [[B16]]:
// LLVM:   %[[V17]] = load i32, ptr {{.*}}, align 4
// LLVM:   %[[V18]] = add nsw i32 %[[V17]], 1
// LLVM:   store i32 %[[V18]], ptr {{.*}}, align 4
// LLVM:   br label {{.*}}
// LLVM: %[[B19:.*]]
// LLVM:   br label %[[B20:.*]]
// LLVM: [[B20]]:
// LLVM:   invoke void @_ZN1SC2Ev(ptr {{.*}})
// LLVM:           to label %[[B21:.*]] unwind label %[[B30:.*]]
// LLVM: [[B21]]:
// LLVM:   br label {{.*}}
// LLVM: [[B22]]:
// LLVM:   %[[V23:.*]] = landingpad { ptr, i32 }
// LLVM:           catch ptr null
// LLVM:   %[[V24:.*]] = extractvalue { ptr, i32 } %[[V23]], 0
// LLVM:   %[[V25:.*]] = extractvalue { ptr, i32 } %[[V23]], 1
// LLVM:   br label %[[B34:.*]]
// LLVM: [[B26]]:
// LLVM:   %[[V27:.*]] = landingpad { ptr, i32 }
// LLVM:           catch ptr null
// LLVM:   %[[V28:.*]] = extractvalue { ptr, i32 } %[[V27]], 0
// LLVM:   %[[V29:.*]] = extractvalue { ptr, i32 } %[[V27]], 1
// LLVM:   br label %[[B34:.*]]
// LLVM: [[B30]]:
// LLVM:   %[[V31:.*]] = landingpad { ptr, i32 }
// LLVM:           catch ptr null
// LLVM:   %[[V32:.*]] = extractvalue { ptr, i32 } %[[V31]], 0
// LLVM:   %[[V33:.*]] = extractvalue { ptr, i32 } %[[V31]], 1
// LLVM:   br label %[[B34:.*]]
// LLVM: [[B34]]:
// LLVM:   %[[V35:.*]] = phi ptr [ %[[V32]], %[[B30]] ], [ %[[V24]], %[[B22]] ], [ %[[V28]], %[[B26]] ]
// LLVM:   %[[V36:.*]] = call ptr @__cxa_begin_catch(ptr %[[V35]])
// LLVM:   %[[V37:.*]] = load i32, ptr {{.*}}, align 4
// LLVM:   %[[V38:.*]] = add nsw i32 %[[V37]], 1
// LLVM:   store i32 %[[V38]], ptr {{.*}}, align 4
// LLVM:   call void @__cxa_end_catch()
// LLVM:   br label {{.*}}

void refoo3() {
  int r = 1;
  try {
    throw;
    S s;
  } catch (...) {
    ++r;
  }
}

// CIR-LABEL: @_Z6refoo3v()
// CIR:   %[[V0:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["r", init] {alignment = 4 : i64}
// CIR:   %[[V1:.*]] = cir.const #cir.int<1> : !s32i
// CIR:   cir.store{{.*}} %[[V1]], %[[V0]] : !s32i, !cir.ptr<!s32i>
// CIR:   cir.scope {
// CIR:     %[[V2:.*]] = cir.alloca !rec_S, !cir.ptr<!rec_S>, ["s", init] {alignment = 1 : i64}
// CIR:     cir.try {
// CIR:       cir.call exception @__cxa_rethrow() : () -> ()
// CIR:       cir.unreachable
// CIR:     ^bb1:  // no predecessors
// CIR:       cir.call exception @_ZN1SC2Ev(%[[V2]]) : (!cir.ptr<!rec_S>) -> ()
// CIR:       cir.yield
// CIR:     } catch [type #cir.all {
// CIR:       %[[V3:.*]] = cir.catch_param -> !cir.ptr<!void>
// CIR:       %[[V4:.*]] = cir.load{{.*}} %[[V0]] : !cir.ptr<!s32i>, !s32i
// CIR:       %[[V5:.*]] = cir.unary(inc, %[[V4]]) nsw : !s32i, !s32i
// CIR:       cir.store{{.*}} %[[V5]], %[[V0]] : !s32i, !cir.ptr<!s32i>
// CIR:       cir.yield
// CIR:     }]
// CIR:   }
// CIR:   cir.return
// CIR: }

// LLVM:  invoke void @__cxa_rethrow()
// LLVM:          to label %[[B5:.*]] unwind label %[[B8:.*]]
// LLVM: [[B5]]:
// LLVM:  unreachable
// LLVM: [[B6]]:
// LLVM:  invoke void @_ZN1SC2Ev(ptr {{.*}})
// LLVM:          to label %[[B7:.*]] unwind label %[[B12:.*]]
// LLVM: [[B7]]:
// LLVM:  br label %[[B21:.*]]
// LLVM: [[B8]]:
// LLVM:  %[[V9:.*]] = landingpad { ptr, i32 }
// LLVM:          catch ptr null
// LLVM:  %[[V10:.*]] = extractvalue { ptr, i32 } %[[V9]], 0
// LLVM:  %[[V11:.*]] = extractvalue { ptr, i32 } %[[V9]], 1
// LLVM:  br label %[[B16:.*]]
// LLVM: [[B12]]:
// LLVM:  %[[V13:.*]] = landingpad { ptr, i32 }
// LLVM:          catch ptr null
// LLVM:  %[[V14:.*]] = extractvalue { ptr, i32 } %[[V13]], 0
// LLVM:  %[[V15:.*]] = extractvalue { ptr, i32 } %[[V13]], 1
// LLVM:  br label %[[B16]]
// LLVM: [[B16]]:
// LLVM:  %[[V17:.*]] = phi ptr [ %[[V14]], %[[B12]] ], [ %[[V10]], %[[B8]] ]
// LLVM:  %[[V18:.*]] = call ptr @__cxa_begin_catch(ptr %[[V17]])
// LLVM:  %[[V19:.*]] = load i32, ptr {{.*}}, align 4
// LLVM:  %[[V20:.*]] = add nsw i32 %[[V19]], 1
// LLVM:  store i32 %[[V20]], ptr {{.*}}, align 4
// LLVM:  call void @__cxa_end_catch()
// LLVM:  br label %[[B21]]
// LLVM: [[B21]]:
// LLVM:  br label {{.*}}

void refoo4() {
  try {
    for (int i = 0; i < 5; i++) {
      throw;
      throw;
      S s;
      i++;
    }
  } catch (...) {
    int r = 1;
  }
}

// CIR-LABEL: @_Z6refoo4v
// CIR: cir.call exception @__cxa_rethrow() : () -> ()
// CIR-NEXT: unreachable
// CIR: cir.call exception @__cxa_rethrow() : () -> ()
// CIR-NEXT: unreachable
// CIR: cir.call exception @_ZN1SC2Ev

// LLVM: invoke void @__cxa_rethrow
// LLVM: unreachable
// LLVM: invoke void @__cxa_rethrow
// LLVM: unreachable
// LLVM: invoke void @_ZN1SC2Ev

void statements() {
  throw 0;
  123 + 456;
}

// CIR:      cir.func {{.*}} @_Z10statementsv()
// CIR-NEXT:   %[[V0:.*]] = cir.alloc.exception 4 -> !cir.ptr<!s32i>
// CIR-NEXT:   %[[V1:.*]] = cir.const #cir.int<0> : !s32i
// CIR-NEXT:   cir.store align(16) %[[V1]], %[[V0]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:   cir.throw %[[V0]] : !cir.ptr<!s32i>, @_ZTIi
// CIR-NEXT:   cir.unreachable
// CIR-NEXT: ^bb1:
// CIR-NEXT:   %[[V2:.*]] = cir.const #cir.int<123> : !s32i
// CIR-NEXT:   %[[V3:.*]] = cir.const #cir.int<456> : !s32i
// CIR-NEXT:   %[[V4:.*]] = cir.binop(add, %[[V2]], %[[V3]]) nsw : !s32i
// CIR-NEXT:   cir.return
// CIR-NEXT: }

// LLVM: call void @__cxa_throw
// LLVM: unreachable

void paren_expr() { (throw 0, 123 + 456); }

// CIR:       cir.func {{.*}} @_Z10paren_exprv()
// CIR-NEXT:   %[[V0:.*]] = cir.alloc.exception 4 -> !cir.ptr<!s32i>
// CIR-NEXT:   %[[V1:.*]] = cir.const #cir.int<0> : !s32i
// CIR-NEXT:   cir.store align(16) %[[V1]], %[[V0]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:   cir.throw %[[V0]] : !cir.ptr<!s32i>, @_ZTIi
// CIR-NEXT:   cir.unreachable
// CIR-NEXT: ^bb1:
// CIR-NEXT:   %[[V2:.*]] = cir.const #cir.int<123> : !s32i
// CIR-NEXT:   %[[V3:.*]] = cir.const #cir.int<456> : !s32i
// CIR-NEXT:   %[[V4:.*]] = cir.binop(add, %[[V2]], %[[V3]]) nsw : !s32i
// CIR-NEXT:   cir.return
// CIR-NEXT: }

// LLVM: call void @__cxa_throw
// LLVM: unreachable

int ternary_throw1(bool condition, int x) {
  return condition ? throw x : x;
}

// CIR:     cir.func {{.*}} @_Z14ternary_throw1bi(%arg0: !cir.bool
// CIR-NEXT:   %[[V0:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["condition", init] {alignment = 1 : i64}
// CIR-NEXT:   %[[V1:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init] {alignment = 4 : i64}
// CIR-NEXT:   %[[V2:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
// CIR-NEXT:   %[[V3:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["cleanup.cond"] {alignment = 1 : i64}
// CIR-NEXT:   %[[V4:.*]] = cir.const #false
// CIR-NEXT:   %[[V5:.*]] = cir.const #true
// CIR-NEXT:   cir.store %arg0, %[[V0]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR-NEXT:   cir.store %arg1, %[[V1]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:   %[[V6:.*]] = cir.load align(1) %[[V0]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR-NEXT:   cir.store align(1) %[[V4]], %[[V3]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR-NEXT:   %[[V7:.*]] = cir.ternary(%[[V6]], true {
// CIR-NEXT:     %[[V9:.*]] = cir.alloc.exception 4 -> !cir.ptr<!s32i>
// CIR-NEXT:     cir.store align(1) %[[V5]], %[[V3]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR-NEXT:     %[[V10:.*]] = cir.load align(4) %[[V1]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:     cir.store align(16) %[[V10]], %[[V9]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:     cir.throw %[[V9]] : !cir.ptr<!s32i>, @_ZTIi
// CIR-NEXT:     cir.unreachable
// CIR-NEXT:   ^bb1:  // no predecessors
// CIR-NEXT:     %[[V11:.*]] = cir.const #cir.int<0> : !s32i loc(#loc173)
// CIR-NEXT:     cir.yield %[[V11]] : !s32i
// CIR-NEXT:   }, false {
// CIR-NEXT:     %[[V9:.*]] = cir.load align(4) %[[V1]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:     cir.yield %[[V9]] : !s32i
// CIR-NEXT:   }) : (!cir.bool) -> !s32i
// CIR-NEXT:   cir.store{{.*}} %[[V7]], %[[V2]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:   %[[V8:.*]] = cir.load{{.*}} %[[V2]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:   cir.return %[[V8]] : !s32i
// CIR-NEXT: }

// LLVM: @_Z14ternary_throw1bi
// LLVM:   %[[V3:.*]] = alloca i8, i64 1, align 1
// LLVM:   %[[V4:.*]] = alloca i32, i64 1, align 4
// LLVM:   %[[V5:.*]] = alloca i32, i64 1, align 4
// LLVM:   %[[V6:.*]] = alloca i8, i64 1, align 1
// LLVM:   %[[V7:.*]] = zext i1 %[[V0:.*]] to i8
// LLVM:   store i8 %[[V7]], ptr %[[V3]], align 1
// LLVM:   store i32 %[[V1:.*]], ptr %[[V4]], align 4
// LLVM:   %[[V8:.*]] = load i8, ptr %[[V3]], align 1
// LLVM:   %[[V9:.*]] = trunc i8 %[[V8]] to i1
// LLVM:   store i8 0, ptr %[[V6]], align 1
// LLVM:   br i1 %[[V9]], label %[[B10:.*]], label %[[B14:.*]]
// LLVM: [[B10]]:
// LLVM:   %[[V11:.*]] = call ptr @__cxa_allocate_exception(i64 4)
// LLVM:   store i8 1, ptr %[[V6]], align 1
// LLVM:   %[[V12:.*]] = load i32, ptr %[[V4]], align 4
// LLVM:   store i32 %[[V12]], ptr %[[V11]], align 16
// LLVM:   call void @__cxa_throw(ptr %[[V11]], ptr @_ZTIi, ptr null)
// LLVM:   unreachable
// LLVM: [[B13]]:
// LLVM:   br label %[[B16:.*]]
// LLVM: [[B14]]:
// LLVM:   %[[V15:.*]] = load i32, ptr %[[V4]], align 4
// LLVM:   br label %[[B16]]
// LLVM: [[B16]]:
// LLVM:   %[[V17:.*]] = phi i32 [ 0, %[[V13]] ], [ %[[V15]], %[[V14]] ]
// LLVM:   store i32 %[[V17]], ptr %[[V5]], align 4
// LLVM:   %[[V18:.*]] = load i32, ptr %[[V5]], align 4
// LLVM:   ret i32 %[[V18]]

int ternary_throw2(bool condition, int x) {
  return condition ? x : throw x;
}

// LLVM: @_Z14ternary_throw2bi
// LLVM:   %[[V3:.*]] = alloca i8, i64 1, align 1
// LLVM:   %[[V4:.*]] = alloca i32, i64 1, align 4
// LLVM:   %[[V5:.*]] = alloca i32, i64 1, align 4
// LLVM:   %[[V6:.*]] = alloca i8, i64 1, align 1
// LLVM:   %[[V7:.*]] = zext i1 %[[V0:.*]] to i8
// LLVM:   store i8 %[[V7]], ptr %[[V3]], align 1
// LLVM:   store i32 %[[V1]], ptr %[[V4]], align 4
// LLVM:   %[[V8:.*]] = load i8, ptr %[[V3]], align 1
// LLVM:   %[[V9:.*]] = trunc i8 %[[V8]] to i1
// LLVM:   store i8 0, ptr %[[V6]], align 1
// LLVM:   br i1 %[[V9]], label %[[B10:.*]], label %[[B12:.*]]
// LLVM: [[B10]]:
// LLVM:   %[[V11:.*]] = load i32, ptr %[[V4]], align 4
// LLVM:   br label %[[B16:.*]]
// LLVM: [[B12]]:
// LLVM:   %[[V13:.*]] = call ptr @__cxa_allocate_exception(i64 4)
// LLVM:   store i8 1, ptr %[[V6]], align 1
// LLVM:   %[[V14:.*]] = load i32, ptr %[[V4]], align 4
// LLVM:   store i32 %[[V14]], ptr %[[V13]], align 16
// LLVM:   call void @__cxa_throw(ptr %[[V13]], ptr @_ZTIi, ptr null)
// LLVM:   unreachable
// LLVM: [[B15:.*]]:
// LLVM:   br label %[[B16:.*]]
// LLVM: [[B16]]:
// LLVM:   %[[V17:.*]] = phi i32 [ 0, %[[V15]] ], [ %[[V11]], %[[V10]] ]
// LLVM:   store i32 %[[V17]], ptr %[[V5]], align 4
// LLVM:   %[[V18:.*]] = load i32, ptr %[[V5]], align 4
// LLVM:   ret i32 %[[V18]]
