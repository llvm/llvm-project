// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fcxx-exceptions -fexceptions -mconstructor-aliases -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fcxx-exceptions -fexceptions -mconstructor-aliases -fclangir -emit-cir-flat -fno-clangir-call-conv-lowering %s -o %t.flat.cir
// RUN: FileCheck --input-file=%t.flat.cir --check-prefix=CIR_FLAT %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fcxx-exceptions -fexceptions -mconstructor-aliases -fclangir -emit-llvm -fno-clangir-call-conv-lowering %s -o %t.ll
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

// CIR-DAG: ![[VecTy:.*]] = !cir.record<struct "Vec" padded {!u8i}>
// CIR-DAG: ![[S1:.*]] = !cir.record<struct "S1" {![[VecTy]]}>

// CIR_FLAT-DAG: ![[VecTy:.*]] = !cir.record<struct "Vec" padded {!u8i}>
// CIR_FLAT-DAG: ![[S1:.*]] = !cir.record<struct "S1" {![[VecTy]]}>

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

// LLVM:   %[[Vec:.*]] = alloca %struct.Vec
// LLVM:   br label %[[INVOKE_BB:.*]]

// LLVM: [[INVOKE_BB]]:
// LLVM:   invoke void @_ZN3VecC1Ev(ptr %[[Vec]])
// LLVM:           to label %[[DTOR_BB:.*]] unwind label %[[LPAD_BB:.*]]

// LLVM: [[DTOR_BB]]:
// LLVM:   call void @_ZN3VecD1Ev(ptr %[[Vec]])
// LLVM:   br label %15

// LLVM: [[LPAD_BB]]:
// LLVM:   landingpad { ptr, i32 }
// LLVM:           catch ptr null
// LLVM:   br label %[[CATCH_BB:.*]]

// LLVM: [[CATCH_BB]]:
// LLVM:   call ptr @__cxa_begin_catch
// LLVM:   call void @__cxa_end_catch()
// LLVM:   br label %[[RET_BB:.*]]

// LLVM: [[RET_BB]]:
// LLVM:   ret void

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
// CIR_FLAT:    cir.try_call @_ZN3VecC1Ev(%[[vec:.+]]) ^[[NEXT_CALL_PREP:.*]], ^[[PAD_NODTOR:.*]] : (!cir.ptr<![[VecTy]]>) -> ()
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
// CIR_FLAT:    cir.call @_ZN3VecD1Ev(%[[vec]]) : (!cir.ptr<![[VecTy]]>) -> ()
// CIR_FLAT:    cir.br ^[[CATCH_BEGIN]](%exception_ptr_0 : !cir.ptr<!void>)
// CIR_FLAT:  ^[[CATCH_BEGIN]](
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

// LLVM-LABEL: @_Z3yo3b
// LLVM:   %[[V1:.*]] = alloca %struct.Vec
// LLVM:   %[[V2:.*]] = alloca %struct.Vec
// LLVM:   %[[V3:.*]] = alloca %struct.Vec
// LLVM:   %[[V4:.*]] = alloca %struct.Vec
// LLVM:   br label %[[CALL0:.*]]
// LLVM: [[CALL0]]:
// LLVM:   invoke void @_ZN3VecC1Ev(ptr %[[V1]])
// LLVM:           to label %[[CALL1:.*]] unwind label %[[LPAD0:.*]]
// LLVM: [[CALL1]]:
// LLVM:   invoke void @_ZN3VecC1Ev(ptr %[[V2]])
// LLVM:           to label %[[CALL2:.*]] unwind label %[[LPAD1:.*]]
// LLVM: [[CALL2]]:
// LLVM:   invoke void @_ZN3VecC1Ev(ptr %[[V3]])
// LLVM:           to label %[[CALL3:.*]] unwind label %[[LPAD2:.*]]
// LLVM: [[CALL3]]:
// LLVM:   invoke void @_ZN3VecC1Ev(ptr %[[V4]])
// LLVM:           to label %[[REGULAR_CLEANUP:.*]] unwind label %[[LPAD3:.*]]
// LLVM: [[REGULAR_CLEANUP]]:
// LLVM:   call void @_ZN3VecD1Ev(ptr %[[V4]])
// LLVM:   call void @_ZN3VecD1Ev(ptr %[[V3]])
// LLVM:   call void @_ZN3VecD1Ev(ptr %[[V2]])
// LLVM:   call void @_ZN3VecD1Ev(ptr %[[V1]])
// LLVM:   br label %[[RET:.*]]
// LLVM: [[LPAD0]]:
// LLVM:   landingpad { ptr, i32 }
// LLVM:           catch ptr null
// LLVM:   br label %[[CATCH:.*]]
// LLVM: [[LPAD1]]:
// LLVM:   landingpad { ptr, i32 }
// LLVM:           catch ptr null
// LLVM:   call void @_ZN3VecD1Ev(ptr %[[V1]])
// LLVM:   br label %[[CATCH]]
// LLVM: [[LPAD2]]:
// LLVM:   landingpad { ptr, i32 }
// LLVM:           catch ptr null
// LLVM:   call void @_ZN3VecD1Ev(ptr %[[V2]])
// LLVM:   call void @_ZN3VecD1Ev(ptr %[[V1]])
// LLVM:   br label %[[CATCH]]
// LLVM: [[LPAD3]]:
// LLVM:   landingpad { ptr, i32 }
// LLVM:           catch ptr null
// LLVM:   call void @_ZN3VecD1Ev(ptr %[[V3]])
// LLVM:   call void @_ZN3VecD1Ev(ptr %[[V2]])
// LLVM:   call void @_ZN3VecD1Ev(ptr %[[V1]])
// LLVM:   br label %[[CATCH]]
// LLVM: [[CATCH]]:
// LLVM:   call ptr @__cxa_begin_catch
// LLVM:   br label %[[RET]]
// LLVM: [[RET]]:
// LLVM:   ret void

void yo2(bool x) {
  int r = 1;
  try {
    Vec v1, v2;
    try {
        Vec v3, v4;
    } catch (...) {
    r++;
    }
  } catch (...) {
    r++;
  }
}

// CIR: cir.scope {
// CIR:   %[[V1:.*]] = cir.alloca ![[VecTy]], !cir.ptr<![[VecTy]]>, ["v1"
// CIR:   %[[V2:.*]] = cir.alloca ![[VecTy]], !cir.ptr<![[VecTy]]>, ["v2"
// CIR:   cir.try {
// CIR:     cir.call exception @_ZN3VecC1Ev(%[[V1]]) : (!cir.ptr<![[VecTy]]>) -> ()
// CIR:     cir.call exception @_ZN3VecC1Ev(%[[V2]]) : (!cir.ptr<![[VecTy]]>) -> () cleanup {
// CIR:       cir.call @_ZN3VecD1Ev(%[[V1]]) : (!cir.ptr<![[VecTy]]>) -> ()
// CIR:       cir.yield
// CIR:     }
// CIR:     cir.scope {
// CIR:       %[[V3:.*]] = cir.alloca ![[VecTy]], !cir.ptr<![[VecTy]]>, ["v3"
// CIR:       %[[V4:.*]] = cir.alloca ![[VecTy]], !cir.ptr<![[VecTy]]>, ["v4"
// CIR:       cir.try {
// CIR:         cir.call exception @_ZN3VecC1Ev(%[[V3]]) : (!cir.ptr<![[VecTy]]>) -> ()
// CIR:         cir.call exception @_ZN3VecC1Ev(%[[V4]]) : (!cir.ptr<![[VecTy]]>) -> () cleanup {
// CIR:           cir.call @_ZN3VecD1Ev(%[[V3]]) : (!cir.ptr<![[VecTy]]>) -> ()
// CIR:           cir.yield
// CIR:         }
// CIR:         cir.call @_ZN3VecD1Ev(%[[V4]]) : (!cir.ptr<![[VecTy]]>) -> ()
// CIR:         cir.call @_ZN3VecD1Ev(%[[V3]]) : (!cir.ptr<![[VecTy]]>) -> ()
// CIR:         cir.yield
// CIR:       } catch [type #cir.all {
// CIR:         cir.catch_param -> !cir.ptr<!void>
// CIR:       }]
// CIR:     }
// CIR:     cir.call @_ZN3VecD1Ev(%[[V2]]) : (!cir.ptr<![[VecTy]]>) -> ()
// CIR:     cir.call @_ZN3VecD1Ev(%[[V1]]) : (!cir.ptr<![[VecTy]]>) -> ()
// CIR:     cir.yield
// CIR:   } catch [type #cir.all {
// CIR:     cir.catch_param -> !cir.ptr<!void>
// CIR:   }]


int foo() { return 42; }

struct A {
  ~A() {}
};

void bar() {
  A a;
  int b = foo();
}

// CIR-LABEL: @_Z3barv
// CIR:  %[[V0:.*]] = cir.alloca !rec_A, !cir.ptr<!rec_A>, ["a"] {alignment = 1 : i64}
// CIR:  %[[V1:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["b", init] {alignment = 4 : i64}
// CIR:  %[[V2:.*]] = cir.call @_Z3foov() : () -> !s32i
// CIR:  cir.store align(4) %[[V2]], %[[V1]] : !s32i, !cir.ptr<!s32i>
// CIR:  cir.call @_ZN1AD2Ev(%[[V0]]) : (!cir.ptr<!rec_A>) -> ()
// CIR:  cir.return

// LLVM: ; Function Attrs: noinline nounwind optnone
// LLVM-NEXT: _Z3foo
// LLVM: @_Z3barv()
// LLVM:   %[[V1:.*]] = alloca %struct.A, i64 1, align 1
// LLVM:   %[[V2:.*]] = alloca i32, i64 1, align 4
// LLVM:   %[[V3:.*]] = call i32 @_Z3foov()
// LLVM:   store i32 %[[V3]], ptr %[[V2]], align 4
// LLVM:   call void @_ZN1AD2Ev(ptr %[[V1]])
// LLVM:   ret void

class C {
public:
  ~C();
  void operator=(C);
};

void d() {
  C a, b;
  a = b;
}

// CIR: %[[V0:.*]] = cir.alloca !rec_C, !cir.ptr<!rec_C>, ["a"] {alignment = 1 : i64}
// CIR-NEXT: %[[V1:.*]] = cir.alloca !rec_C, !cir.ptr<!rec_C>, ["b"] {alignment = 1 : i64}
// CIR-NEXT: cir.scope {
// CIR-NEXT:   %[[V2:.*]] = cir.alloca !rec_C, !cir.ptr<!rec_C>, ["agg.tmp0"] {alignment = 1 : i64}
// CIR-NEXT:   cir.copy %[[V1]] to %[[V2]] : !cir.ptr<!rec_C>
// CIR-NEXT:   %[[V3:.*]] = cir.load{{.*}} %[[V2]] : !cir.ptr<!rec_C>, !rec_C
// CIR-NEXT:   cir.try synthetic cleanup {
// CIR-NEXT:     cir.call exception @_ZN1CaSES_(%[[V0]], %[[V3]]) : (!cir.ptr<!rec_C>, !rec_C) -> () cleanup {
// CIR-NEXT:       cir.call @_ZN1CD1Ev(%[[V2]]) : (!cir.ptr<!rec_C>) -> ()
// CIR-NEXT:       cir.call @_ZN1CD1Ev(%[[V1]]) : (!cir.ptr<!rec_C>) -> ()
// CIR-NEXT:       cir.yield
// CIR-NEXT:     }
// CIR-NEXT:     cir.yield
// CIR-NEXT:   } catch [#cir.unwind {
// CIR-NEXT:     cir.resume
// CIR-NEXT:   }]
// CIR-NEXT:   cir.call @_ZN1CD1Ev(%[[V2]]) : (!cir.ptr<!rec_C>) -> ()
// CIR-NEXT: }
// CIR-NEXT: cir.call @_ZN1CD1Ev(%[[V1]]) : (!cir.ptr<!rec_C>) -> ()
// CIR-NEXT: cir.call @_ZN1CD1Ev(%[[V0]]) : (!cir.ptr<!rec_C>) -> ()
// CIR-NEXT: cir.return

template <typename> class a;

template <> class a<void> {
public:
  struct b {
    typedef a<int> c;
  };
};

template <typename> class a {
public:
  template <typename d> a(d) noexcept;
  ~a();
};

struct e {
  using f = a<void>::b::c;
};

template <typename, typename> using g = e::f;

template <typename h> void i(h);

class j {

public:
  using k = g<int, j>;
};

class l {
public:
  template <typename m, typename n> l(m p1, n) : l(p1, 0, a<void>()) {}
  template <typename m, typename n, typename h> l(m, n, h o) {
    try {
      j::k p(o);
      i(p);
    } catch (...) {
    }
  }
};

class G {
public:
  template <typename q, typename n> G(q p1, n) : r(p1, 0) {}
  l r;
};

class s : G {
public:
  int t;
  s() : G(t, 0) {}
};

void fn3() { s(); }

// CIR: cir.func {{.*}} @_ZN1lC2Iii1aIvEEET_T0_T1_
// CIR:   cir.scope
// CIR:     %[[V5:.*]] = cir.alloca !rec_a3Cint3E, !cir.ptr<!rec_a3Cint3E>
// CIR:     %[[V6:.*]] = cir.alloca !rec_a3Cvoid3E, !cir.ptr<!rec_a3Cvoid3E>
// CIR:     cir.try {
// CIR:       cir.copy {{.*}} to %[[V6]] : !cir.ptr<!rec_a3Cvoid3E>
// CIR:       %[[V7:.*]] = cir.load align(1) %[[V6]] : !cir.ptr<!rec_a3Cvoid3E>, !rec_a3Cvoid3E
// CIR:       cir.call @_ZN1aIiEC1IS_IvEEET_(%[[V5]], %[[V7]]) : (!cir.ptr<!rec_a3Cint3E>, !rec_a3Cvoid3E) -> ()
// CIR:       cir.scope {
// CIR:         %[[V8:.*]] = cir.alloca !rec_a3Cint3E, !cir.ptr<!rec_a3Cint3E>
// CIR:         cir.copy %[[V5]] to %[[V8]] : !cir.ptr<!rec_a3Cint3E>
// CIR:         %[[V9:.*]] = cir.load align(1) %[[V8]] : !cir.ptr<!rec_a3Cint3E>, !rec_a3Cint3E
// CIR-NEXT:         cir.call exception @_Z1iI1aIiEEvT_(%[[V9]]) : (!rec_a3Cint3E) -> () cleanup {
// CIR-NEXT:           cir.call @_ZN1aIiED1Ev(%[[V8]]) : (!cir.ptr<!rec_a3Cint3E>) -> ()
// CIR-NEXT:           cir.call @_ZN1aIiED1Ev(%[[V5]]) : (!cir.ptr<!rec_a3Cint3E>) -> ()
// CIR-NEXT:           cir.yield
// CIR-NEXT:         }
// CIR-NEXT:         cir.call @_ZN1aIiED1Ev(%[[V8]]) : (!cir.ptr<!rec_a3Cint3E>) -> ()
// CIR-NEXT:       }
// CIR-NEXT:       cir.call @_ZN1aIiED1Ev(%[[V5]]) : (!cir.ptr<!rec_a3Cint3E>) -> ()
// CIR-NEXT:       cir.yield
// CIR:     } catch [type #cir.all {
// CIR:       %[[V7:.*]] = cir.catch_param -> !cir.ptr<!void>
// CIR:       cir.yield
// CIR:     }]
