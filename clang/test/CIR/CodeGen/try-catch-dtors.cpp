// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -mconstructor-aliases -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -mconstructor-aliases -fclangir -emit-llvm %s -o %t.ll
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

// CIR: cir.scope {
// CIR:   %[[VADDR:.*]] = cir.alloca ![[VecTy]], !cir.ptr<![[VecTy]]>, ["v", init]
// CIR:   cir.try {
// CIR:     cir.call exception @_ZN3VecC1Ev(%[[VADDR]]) : (!cir.ptr<![[VecTy]]>) -> ()
// CIR:     cir.call @_ZN3VecD1Ev(%[[VADDR]]) : (!cir.ptr<![[VecTy]]>) -> ()
// CIR:     cir.yield
// CIR:   } cleanup {
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