// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fcxx-exceptions -fexceptions -mconstructor-aliases -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fcxx-exceptions -fexceptions -mconstructor-aliases -fclangir -emit-cir-flat %s -o %t.flat.cir
// RUN: FileCheck --input-file=%t.flat.cir --check-prefix=CIR_FLAT %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -DLLVM_IMPLEMENTED -fcxx-exceptions -fexceptions -mconstructor-aliases -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

struct Vec {
  Vec();
  Vec(Vec&&);
  ~Vec();
};

void yo() {
  int r = 1;
  try {
    Vec v;
  } catch (...) {
    r++;
  }
}

// CIR-DAG: ![[VecTy:.*]] = !cir.struct<struct "Vec" {!cir.int<u, 8>}>
// CIR-DAG: ![[S1:.*]] = !cir.struct<struct "S1" {!cir.struct<struct "Vec" {!cir.int<u, 8>}>}>

// CIR_FLAT-DAG: ![[VecTy:.*]] = !cir.struct<struct "Vec" {!cir.int<u, 8>}>
// CIR_FLAT-DAG: ![[S1:.*]] = !cir.struct<struct "S1" {!cir.struct<struct "Vec" {!cir.int<u, 8>}>}>

// CIR: cir.scope {
// CIR:   %[[VADDR:.*]] = cir.alloca ![[VecTy]], !cir.ptr<![[VecTy]]>, ["v", init]
// CIR:   cir.try {
// CIR:     cir.call exception @_ZN3VecC1Ev(%[[VADDR]]) : (!cir.ptr<![[VecTy]]>) -> ()
// CIR:     cir.call @_ZN3VecD1Ev(%[[VADDR]]) : (!cir.ptr<![[VecTy]]>) -> ()
// CIR:     cir.yield
// CIR:   } catch [type #cir.all {
// CIR:     cir.catch_param -> !cir.ptr<!void>
// CIR:   }]
// CIR: }
// CIR: cir.return

// LLVM-LABEL: @_Z2yov()

// LLVM: 2:
// LLVM:   %[[Vec:.*]] = alloca %struct.Vec
// LLVM:   br label %[[INVOKE_BB:.*]],

// LLVM: [[INVOKE_BB]]:
// LLVM:   invoke void @_ZN3VecC1Ev(ptr %[[Vec]])
// LLVM:           to label %[[DTOR_BB:.*]] unwind label %[[LPAD_BB:.*]],

// LLVM: [[DTOR_BB]]:
// LLVM:   call void @_ZN3VecD1Ev(ptr %[[Vec]])
// LLVM:   br label %15

// LLVM: [[LPAD_BB]]:
// LLVM:   landingpad { ptr, i32 }
// LLVM:           catch ptr null
// LLVM:   br label %[[CATCH_BB:.*]],

// LLVM: [[CATCH_BB]]:
// LLVM:   call ptr @__cxa_begin_catch
// LLVM:   call void @__cxa_end_catch()
// LLVM:   br label %[[RET_BB:.*]],

// LLVM: [[RET_BB]]:
// LLVM:   ret void

#ifndef LLVM_IMPLEMENTED
struct S1 {
  Vec v;
};

void yo2() {
  int r = 1;
  try {
    Vec v;
    S1((Vec&&) v);
  } catch (...) {
    r++;
  }
}
// CIR-LABEL: @_Z3yo2v
// CIR:   cir.scope {
// CIR:     cir.alloca ![[VecTy]]
// CIR:     cir.try {
// CIR:       cir.call exception @_ZN3VecC1Ev
// CIR:       cir.scope {
// CIR:         cir.alloca ![[S1:.*]], !cir.ptr<![[S1:.*]]>, ["agg.tmp.ensured"]
// CIR:         cir.call exception @_ZN3VecC1EOS_{{.*}} cleanup {
// CIR:           cir.call @_ZN3VecD1Ev
// CIR:           cir.yield
// CIR:         cir.call @_ZN2S1D2Ev
// CIR:       }
// CIR:       cir.call @_ZN3VecD1Ev
// CIR:       cir.yield
// CIR:     } catch [type #cir.all {
// CIR:       cir.catch_param -> !cir.ptr<!void>
// CIR:       cir.yield
// CIR:     }]
// CIR:   }
// CIR:   cir.return
// CIR: }

// CIR_FLAT-LABEL: @_Z3yo2v
// CIR_FLAT:    cir.try_call @_ZN3VecC1Ev(%2) ^[[NEXT_CALL_PREP:.*]], ^[[PAD_NODTOR:.*]] : (!cir.ptr<![[VecTy]]>) -> ()
// CIR_FLAT:  ^[[NEXT_CALL_PREP]]:
// CIR_FLAT:    cir.br ^[[NEXT_CALL:.*]] loc
// CIR_FLAT:  ^[[NEXT_CALL]]:
// CIR_FLAT:    cir.try_call @_ZN3VecC1EOS_({{.*}}) ^[[CONT0:.*]], ^[[PAD_DTOR:.*]] :
// CIR_FLAT:  ^[[CONT0]]:
// CIR_FLAT:    cir.call @_ZN2S1D2Ev
// CIR_FLAT:    cir.br ^[[CONT1:.*]] loc
// CIR_FLAT:  ^[[CONT1]]:
// CIR_FLAT:    cir.call @_ZN3VecD1Ev
// CIR_FLAT:    cir.br ^[[AFTER_TRY:.*]] loc
// CIR_FLAT:  ^[[PAD_NODTOR]]:
// CIR_FLAT:    %exception_ptr, %type_id = cir.eh.inflight_exception
// CIR_FLAT:    cir.br ^[[CATCH_BEGIN:.*]](%exception_ptr : !cir.ptr<!void>)
// CIR_FLAT:  ^[[PAD_DTOR]]:
// CIR_FLAT:    %exception_ptr_0, %type_id_1 = cir.eh.inflight_exception
// CIR_FLAT:    cir.call @_ZN3VecD1Ev(%2) : (!cir.ptr<![[VecTy]]>) -> ()
// CIR_FLAT:    cir.br ^[[CATCH_BEGIN]](%exception_ptr_0 : !cir.ptr<!void>)
// CIR_FLAT:  ^[[CATCH_BEGIN]](%5: !cir.ptr<!void>
// CIR_FLAT:    cir.catch_param begin
// CIR_FLAT:    cir.br ^[[AFTER_TRY]]
// CIR_FLAT:  ^[[AFTER_TRY]]:
// CIR_FLAT:    cir.return
// CIR_FLAT:  }

void yo3(bool x) {
  int r = 1;
  try {
    Vec v1, v2, v3, v4;
  } catch (...) {
    r++;
  }
}

// CIR-LABEL: @_Z3yo3b
// CIR: cir.scope {
// CIR:   %[[V1:.*]] = cir.alloca ![[VecTy]], !cir.ptr<![[VecTy]]>, ["v1"
// CIR:   %[[V2:.*]] = cir.alloca ![[VecTy]], !cir.ptr<![[VecTy]]>, ["v2"
// CIR:   %[[V3:.*]] = cir.alloca ![[VecTy]], !cir.ptr<![[VecTy]]>, ["v3"
// CIR:   %[[V4:.*]] = cir.alloca ![[VecTy]], !cir.ptr<![[VecTy]]>, ["v4"
// CIR:   cir.try {
// CIR:     cir.call exception @_ZN3VecC1Ev(%[[V1]]) : (!cir.ptr<![[VecTy]]>) -> ()
// CIR:     cir.call exception @_ZN3VecC1Ev(%[[V2]]) : (!cir.ptr<![[VecTy]]>) -> () cleanup {
// CIR:       cir.call @_ZN3VecD1Ev(%[[V1]]) : (!cir.ptr<![[VecTy]]>) -> ()
// CIR:       cir.yield
// CIR:     }
// CIR:     cir.call exception @_ZN3VecC1Ev(%[[V3]]) : (!cir.ptr<![[VecTy]]>) -> () cleanup {
// CIR:       cir.call @_ZN3VecD1Ev(%[[V2]]) : (!cir.ptr<![[VecTy]]>) -> ()
// CIR:       cir.call @_ZN3VecD1Ev(%[[V1]]) : (!cir.ptr<![[VecTy]]>) -> ()
// CIR:       cir.yield
// CIR:     }
// CIR:     cir.call exception @_ZN3VecC1Ev(%[[V4]]) : (!cir.ptr<![[VecTy]]>) -> () cleanup {
// CIR:       cir.call @_ZN3VecD1Ev(%[[V3]]) : (!cir.ptr<![[VecTy]]>) -> ()
// CIR:       cir.call @_ZN3VecD1Ev(%[[V2]]) : (!cir.ptr<![[VecTy]]>) -> ()
// CIR:       cir.call @_ZN3VecD1Ev(%[[V1]]) : (!cir.ptr<![[VecTy]]>) -> ()
// CIR:       cir.yield
// CIR:     }
// CIR:     cir.call @_ZN3VecD1Ev(%[[V4]]) : (!cir.ptr<![[VecTy]]>) -> ()
// CIR:     cir.call @_ZN3VecD1Ev(%[[V3]]) : (!cir.ptr<![[VecTy]]>) -> ()
// CIR:     cir.call @_ZN3VecD1Ev(%[[V2]]) : (!cir.ptr<![[VecTy]]>) -> ()
// CIR:     cir.call @_ZN3VecD1Ev(%[[V1]]) : (!cir.ptr<![[VecTy]]>) -> ()
// CIR:     cir.yield
// CIR:   } catch [type #cir.all {
// CIR:   }]
// CIR: }
// CIR: cir.return

// CIR_FLAT-LABEL: @_Z3yo3b
// CIR_FLAT:  ^bb1:
// CIR_FLAT:   %[[V1:.*]] = cir.alloca ![[VecTy]], !cir.ptr<![[VecTy]]>, ["v1"
// CIR_FLAT:   %[[V2:.*]] = cir.alloca ![[VecTy]], !cir.ptr<![[VecTy]]>, ["v2"
// CIR_FLAT:   %[[V3:.*]] = cir.alloca ![[VecTy]], !cir.ptr<![[VecTy]]>, ["v3"
// CIR_FLAT:   %[[V4:.*]] = cir.alloca ![[VecTy]], !cir.ptr<![[VecTy]]>, ["v4"
// CIR_FLAT:    cir.br ^[[CALL0:.*]] loc
// CIR_FLAT:  ^[[CALL0]]:
// CIR_FLAT:    cir.try_call @_ZN3VecC1Ev(%[[V1]]) ^[[CALL1:.*]], ^[[CLEANUP_V1:.*]] : (!cir.ptr<![[VecTy]]>) -> ()
// CIR_FLAT:  ^[[CALL1]]:
// CIR_FLAT:    cir.try_call @_ZN3VecC1Ev(%[[V2]]) ^[[CALL2:.*]], ^[[CLEANUP_V2:.*]] : (!cir.ptr<![[VecTy]]>) -> ()
// CIR_FLAT:  ^[[CALL2]]:
// CIR_FLAT:    cir.try_call @_ZN3VecC1Ev(%[[V3]]) ^[[CALL3:.*]], ^[[CLEANUP_V3:.*]] : (!cir.ptr<![[VecTy]]>) -> ()
// CIR_FLAT:  ^[[CALL3]]:
// CIR_FLAT:    cir.try_call @_ZN3VecC1Ev(%[[V4]]) ^[[NOTROW_CLEANUP:.*]], ^[[CLEANUP_V4:.*]] : (!cir.ptr<![[VecTy]]>) -> ()
// CIR_FLAT:  ^[[NOTROW_CLEANUP]]:
// CIR_FLAT:    cir.call @_ZN3VecD1Ev(%[[V4]]) : (!cir.ptr<![[VecTy]]>) -> ()
// CIR_FLAT:    cir.call @_ZN3VecD1Ev(%[[V3]]) : (!cir.ptr<![[VecTy]]>) -> ()
// CIR_FLAT:    cir.call @_ZN3VecD1Ev(%[[V2]]) : (!cir.ptr<![[VecTy]]>) -> ()
// CIR_FLAT:    cir.call @_ZN3VecD1Ev(%[[V1]]) : (!cir.ptr<![[VecTy]]>) -> ()
// CIR_FLAT:    cir.br ^[[AFTER_TRY:.*]] loc
// CIR_FLAT:  ^[[CLEANUP_V1]]:
// CIR_FLAT:    %exception_ptr, %type_id = cir.eh.inflight_exception
// CIR_FLAT:    cir.br ^[[CATCH_BEGIN:.*]](%exception_ptr : !cir.ptr<!void>)
// CIR_FLAT:  ^[[CLEANUP_V2]]:
// CIR_FLAT:    %exception_ptr_0, %type_id_1 = cir.eh.inflight_exception
// CIR_FLAT:    cir.call @_ZN3VecD1Ev(%[[V1]]) : (!cir.ptr<![[VecTy]]>) -> ()
// CIR_FLAT:    cir.br ^[[CATCH_BEGIN]](%exception_ptr_0 : !cir.ptr<!void>)
// CIR_FLAT:  ^[[CLEANUP_V3]]:
// CIR_FLAT:    %exception_ptr_2, %type_id_3 = cir.eh.inflight_exception
// CIR_FLAT:    cir.call @_ZN3VecD1Ev(%[[V2]]) : (!cir.ptr<![[VecTy]]>) -> ()
// CIR_FLAT:    cir.call @_ZN3VecD1Ev(%[[V1]]) : (!cir.ptr<![[VecTy]]>) -> ()
// CIR_FLAT:    cir.br ^[[CATCH_BEGIN]](%exception_ptr_2 : !cir.ptr<!void>)
// CIR_FLAT:  ^[[CLEANUP_V4]]:
// CIR_FLAT:    %exception_ptr_4, %type_id_5 = cir.eh.inflight_exception
// CIR_FLAT:    cir.call @_ZN3VecD1Ev(%[[V3]]) : (!cir.ptr<![[VecTy]]>) -> ()
// CIR_FLAT:    cir.call @_ZN3VecD1Ev(%[[V2]]) : (!cir.ptr<![[VecTy]]>) -> ()
// CIR_FLAT:    cir.call @_ZN3VecD1Ev(%[[V1]]) : (!cir.ptr<![[VecTy]]>) -> ()
// CIR_FLAT:    cir.br ^[[CATCH_BEGIN]](%exception_ptr_4 : !cir.ptr<!void>)
// CIR_FLAT:  ^[[CATCH_BEGIN]]({{.*}}
// CIR_FLAT:    cir.catch_param begin
// CIR_FLAT:    cir.br ^[[AFTER_TRY]]
// CIR_FLAT:  ^[[AFTER_TRY]]:
// CIR_FLAT:    cir.return

#endif