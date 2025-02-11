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
// CIR-NEXT:   %[[STR_ADD:.*]] = cir.cast(array_to_ptrdecay, %[[STR]] : !cir.ptr<!cir.array<!s8i x 28>>), !cir.ptr<!s8i>
// CIR-NEXT:   cir.store %[[STR_ADD]], %[[ADDR]] : !cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>>
// CIR-NEXT:   cir.throw %[[ADDR]] : !cir.ptr<!cir.ptr<!s8i>>, @_ZTIPKc
// CIR-NEXT:   cir.unreachable
// CIR-NEXT: }

// LLVM: %[[ADDR:.*]] = call ptr @__cxa_allocate_exception(i64 8)
// LLVM: store ptr @.str, ptr %[[ADDR]], align 8
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
// CIR:   cir.store %[[V1]], %[[V0]] : !s32i, !cir.ptr<!s32i>
// CIR:   cir.scope {
// CIR:     %[[V2:.*]] = cir.alloca !ty_S, !cir.ptr<!ty_S>, ["s", init] {alignment = 1 : i64}
// CIR:     cir.try {
// CIR:       cir.call exception @_ZN1SC2Ev(%[[V2]]) : (!cir.ptr<!ty_S>) -> ()
// CIR:       cir.call exception @__cxa_rethrow() : () -> ()
// CIR:       cir.unreachable
// CIR:     } catch [type #cir.all {
// CIR:       %[[V3:.*]] = cir.catch_param -> !cir.ptr<!void>
// CIR:       %[[V4:.*]] = cir.load %[[V0]] : !cir.ptr<!s32i>, !s32i
// CIR:       %[[V5:.*]] = cir.unary(inc, %[[V4]]) : !s32i, !s32i
// CIR:       cir.store %[[V5]], %[[V0]] : !s32i, !cir.ptr<!s32i>
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
// LLVM:   %[[V19:.*]] = add i32 %[[V18]], 1
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
// CIR:   cir.store %[[V1]], %[[V0]] : !s32i, !cir.ptr<!s32i>
// CIR:   cir.scope {
// CIR:     %[[V2:.*]] = cir.alloca !ty_S, !cir.ptr<!ty_S>, ["s", init] {alignment = 1 : i64}
// CIR:     cir.try {
// CIR:       cir.scope {
// CIR:         %[[V3:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init] {alignment = 4 : i64}
// CIR:         %[[V4:.*]] = cir.const #cir.int<0> : !s32i
// CIR:         cir.store %[[V4]], %[[V3]] : !s32i, !cir.ptr<!s32i>
// CIR:         cir.for : cond {
// CIR:           %[[V5:.*]] = cir.load %[[V3]] : !cir.ptr<!s32i>, !s32i
// CIR:           %[[V6:.*]] = cir.const #cir.int<5> : !s32i
// CIR:           %[[V7:.*]] = cir.cmp(lt, %[[V5]], %[[V6]]) : !s32i, !cir.bool
// CIR:           cir.condition(%[[V7]])
// CIR:         } body {
// CIR:           cir.scope {
// CIR:             %[[V5:.*]] = cir.alloca !ty_S, !cir.ptr<!ty_S>, ["s", init] {alignment = 1 : i64}
// CIR:             cir.call exception @_ZN1SC2Ev(%[[V5]]) : (!cir.ptr<!ty_S>) -> ()
// CIR:             cir.call exception @__cxa_rethrow() : () -> ()
// CIR:             cir.unreachable
// CIR:           }
// CIR:           cir.yield
// CIR:         } step {
// CIR:           %[[V5:.*]] = cir.load %[[V3]] : !cir.ptr<!s32i>, !s32i
// CIR:           %[[V6:.*]] = cir.unary(inc, %[[V5]]) : !s32i, !s32i
// CIR:           cir.store %[[V6]], %[[V3]] : !s32i, !cir.ptr<!s32i>
// CIR:           cir.yield
// CIR:         }
// CIR:       }
// CIR:       cir.call exception @_ZN1SC2Ev(%[[V2]]) : (!cir.ptr<!ty_S>) -> ()
// CIR:       cir.yield
// CIR:     } catch [type #cir.all {
// CIR:       %[[V3:.*]] = cir.catch_param -> !cir.ptr<!void>
// CIR:       %[[V4:.*]] = cir.load %[[V0]] : !cir.ptr<!s32i>, !s32i
// CIR:       %[[V5:.*]] = cir.unary(inc, %[[V4]]) : !s32i, !s32i
// CIR:       cir.store %[[V5]], %[[V0]] : !s32i, !cir.ptr<!s32i>
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
// LLVM:   %[[V18]] = add i32 %[[V17]], 1
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
// LLVM:   %[[V38:.*]] = add i32 %[[V37]], 1
// LLVM:   store i32 %[[V38]], ptr {{.*}}, align 4
// LLVM:   call void @__cxa_end_catch()
// LLVM:   br label {{.*}}
