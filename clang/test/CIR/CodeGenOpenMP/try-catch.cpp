// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fcxx-exceptions -fexceptions -fclangir -emit-cir %s -o %t.parallel.cir
// RUN: FileCheck --input-file=%t.parallel.cir %s --check-prefix=PARALLEL-CIR
// RUN: cir-opt -cir-hoist-allocas -cir-flatten-cfg -cir-eh-abi-lowering %t.parallel.cir -o %t.parallel.eh.cir
// RUN: FileCheck --input-file=%t.parallel.eh.cir %s --check-prefix=PARALLEL-EH --implicit-check-not='!cir.eh_token'
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fcxx-exceptions -fexceptions -fclangir -emit-llvm -DOMIT_PARALLEL_TYPED_CATCH %s -o %t.parallel-cir.ll
// RUN: FileCheck --input-file=%t.parallel-cir.ll %s --check-prefix=PARALLEL-LLVM
// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fcxx-exceptions -fexceptions -fclangir -emit-llvm %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=PARALLEL-TYPED-ERR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fcxx-exceptions -fexceptions -emit-llvm %s -o %t.parallel.ll
// RUN: FileCheck --input-file=%t.parallel.ll %s --check-prefix=PARALLEL-OGCG

void mayThrow();

struct MyException {
  MyException();
  MyException(const MyException &);
  ~MyException();
  int get();
};

// Test whether landing pads are in the omp.parallel region, not in the function body.
// Also test whether EH ABI lowering performs its lowering within those regions.

void catch_all_in_parallel() {
#pragma omp parallel
  {
    try {
      mayThrow();
    } catch (...) {
    }
  }
}

// PARALLEL-CIR-LABEL: cir.func {{.*}} @_Z21catch_all_in_parallelv()
// PARALLEL-CIR:         omp.parallel {
// PARALLEL-CIR:           cir.try {
// PARALLEL-CIR:             cir.call @_Z8mayThrowv()
// PARALLEL-CIR:           } catch all (%{{.*}}: !cir.eh_token {{.*}}) {
// PARALLEL-CIR:           omp.terminator

// PARALLEL-EH-LABEL: cir.func {{.*}} @_Z21catch_all_in_parallelv()
// PARALLEL-EH:         omp.parallel {
// PARALLEL-EH:           cir.try_call @_Z8mayThrowv() ^{{.*}}, ^[[LPAD:bb[0-9]+]]
// PARALLEL-EH:         ^[[LPAD]]:
// PARALLEL-EH:           %[[EXN:.*]], %[[TID:.*]] = cir.eh.inflight_exception catch_all
// PARALLEL-EH:           cir.br ^[[DISPATCH:bb[0-9]+]](%[[EXN]], %[[TID]] : !cir.ptr<!void>, !u32i)
// PARALLEL-EH:         ^[[DISPATCH]](%[[D_EXN:.*]]: !cir.ptr<!void>, %[[D_TID:.*]]: !u32i):
// PARALLEL-EH:           cir.br ^[[CATCH:bb[0-9]+]](%[[D_EXN]], %[[D_TID]] : !cir.ptr<!void>, !u32i)
// PARALLEL-EH:         ^[[CATCH]](%[[C_EXN:.*]]: !cir.ptr<!void>, %{{.*}}: !u32i):
// PARALLEL-EH:           cir.call @__cxa_begin_catch(%[[C_EXN]])
// PARALLEL-EH:           cir.call @__cxa_end_catch()
// PARALLEL-EH:           omp.terminator
// PARALLEL-EH-NEXT:    }
// PARALLEL-EH-NEXT:    cir.return

// PARALLEL-LLVM-LABEL: define dso_local void @_Z21catch_all_in_parallelv()
// PARALLEL-LLVM:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @{{.*}}, i32 0, ptr @_Z21catch_all_in_parallelv..omp_par)

// PARALLEL-LLVM-LABEL: define internal void @_Z21catch_all_in_parallelv..omp_par(
// PARALLEL-LLVM-SAME:    personality ptr @__gxx_personality_v0
// PARALLEL-LLVM:         invoke void @_Z8mayThrowv()
// PARALLEL-LLVM-NEXT:            to label %{{.*}} unwind label %[[LPAD:.*]]
// PARALLEL-LLVM:       [[LPAD]]:
// PARALLEL-LLVM-NEXT:    %{{.*}} = landingpad { ptr, i32 }
// PARALLEL-LLVM-NEXT:            catch ptr null
// PARALLEL-LLVM:         call ptr @__cxa_begin_catch(ptr %{{.*}})
// FIXME: Classic CodeGen invokes __cxa_end_catch with a terminate landing pad.
// PARALLEL-LLVM:         call void @__cxa_end_catch()

// PARALLEL-OGCG-LABEL: define dso_local void @_Z21catch_all_in_parallelv()
// PARALLEL-OGCG:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @{{.*}}, i32 0, ptr @_Z21catch_all_in_parallelv.omp_outlined)

// PARALLEL-OGCG-LABEL: define internal void @_Z21catch_all_in_parallelv.omp_outlined(
// PARALLEL-OGCG-SAME:    personality ptr @__gxx_personality_v0
// PARALLEL-OGCG:         invoke void @_Z8mayThrowv()
// PARALLEL-OGCG-NEXT:            to label %{{.*}} unwind label %[[LPAD:.*]]
// PARALLEL-OGCG:       [[LPAD]]:
// PARALLEL-OGCG-NEXT:    %{{.*}} = landingpad { ptr, i32 }
// PARALLEL-OGCG-NEXT:            catch ptr null
// PARALLEL-OGCG:         call ptr @__cxa_begin_catch(ptr %{{.*}})
// PARALLEL-OGCG:         invoke void @__cxa_end_catch()

// PARALLEL-TYPED-ERR: try-catch.cpp:[[#@LINE+5]]:5: error: catching a specific exception type inside an OpenMP parallel region is not yet implemented
#ifndef OMIT_PARALLEL_TYPED_CATCH
void catch_by_copy_in_parallel() {
#pragma omp parallel
  {
    try {
      mayThrow();
    } catch (MyException e) {
    }
  }
}
#endif

// PARALLEL-CIR-LABEL: cir.func {{.*}} @_Z25catch_by_copy_in_parallelv()
// PARALLEL-CIR:         omp.parallel {
// PARALLEL-CIR:           %[[E:.*]] = cir.alloca "e"
// PARALLEL-CIR:           cir.try {
// PARALLEL-CIR:             cir.call @_Z8mayThrowv()
// PARALLEL-CIR:           } catch [type #cir.global_view<@_ZTI11MyException> : !cir.ptr<!u8i>] (%[[TOK:.*]]: !cir.eh_token {{.*}}) {
// PARALLEL-CIR:             cir.construct_catch_param non_trivial_copy %[[TOK]] to %[[E]] using @__clang_cir_catch_copy__ZTS11MyException
// PARALLEL-CIR:           } unwind (%[[UTOK:.*]]: !cir.eh_token {{.*}}) {
// PARALLEL-CIR:             cir.resume %[[UTOK]] : !cir.eh_token
// PARALLEL-CIR:           omp.terminator

// The copy constructor can throw, so it unwinds to a terminate landing pad,
// which must be in the omp.parallel region with the call.
// PARALLEL-EH-LABEL: cir.func {{.*}} @_Z25catch_by_copy_in_parallelv()
// PARALLEL-EH:         omp.parallel {
// PARALLEL-EH:           %[[E:.*]] = cir.alloca "e"
// PARALLEL-EH:           cir.try_call @_Z8mayThrowv() ^{{.*}}, ^[[LPAD:bb[0-9]+]]
// PARALLEL-EH:         ^[[LPAD]]:
// PARALLEL-EH:           %[[EXN:.*]], %[[TID:.*]] = cir.eh.inflight_exception [@_ZTI11MyException]
// PARALLEL-EH:           cir.br ^[[DISPATCH:bb[0-9]+]](%[[EXN]], %[[TID]] : !cir.ptr<!void>, !u32i)
// PARALLEL-EH:         ^[[DISPATCH]](%[[D_EXN:.*]]: !cir.ptr<!void>, %[[D_TID:.*]]: !u32i):
// PARALLEL-EH:           cir.br ^[[CMP:bb[0-9]+]](%[[D_EXN]], %[[D_TID]] : !cir.ptr<!void>, !u32i)
// PARALLEL-EH:         ^[[CMP]](%[[CMP_EXN:.*]]: !cir.ptr<!void>, %[[CMP_TID:.*]]: !u32i):
// PARALLEL-EH:           %[[TYPEID:.*]] = cir.eh.typeid @_ZTI11MyException
// PARALLEL-EH:           %[[MATCH:.*]] = cir.cmp eq %[[CMP_TID]], %[[TYPEID]] : !u32i
// PARALLEL-EH:           cir.brcond %[[MATCH]] ^[[CATCH:bb[0-9]+]](%[[CMP_EXN]], %[[CMP_TID]] : !cir.ptr<!void>, !u32i), ^[[RESUME:bb[0-9]+]](%[[CMP_EXN]], %[[CMP_TID]] : !cir.ptr<!void>, !u32i)
// PARALLEL-EH:         ^[[CATCH]](%[[C_EXN:.*]]: !cir.ptr<!void>, %{{.*}}: !u32i):
// PARALLEL-EH:           %[[EXN_OBJ:.*]] = cir.call @__cxa_get_exception_ptr(%[[C_EXN]]) nothrow
// PARALLEL-EH:           %[[EXN_E:.*]] = cir.cast bitcast %[[EXN_OBJ]] : !cir.ptr<!u8i> -> !cir.ptr<!rec_MyException>
// PARALLEL-EH:           cir.try_call @_ZN11MyExceptionC1ERKS_(%[[E]], %[[EXN_E]]) ^[[COPIED:bb[0-9]+]], ^[[TERMINATE:bb[0-9]+]]
// PARALLEL-EH:         ^[[COPIED]]:
// PARALLEL-EH:           cir.call @__cxa_begin_catch(%[[C_EXN]])
// PARALLEL-EH:           cir.call @_ZN11MyExceptionD1Ev(%[[E]]) nothrow
// PARALLEL-EH:           cir.call @__cxa_end_catch()
// FIXME: Classic CodeGen calls __clang_call_terminate for an unmatched
// exception instead of resuming it out of the parallel region.
// PARALLEL-EH:         ^[[RESUME]](%[[R_EXN:.*]]: !cir.ptr<!void>, %[[R_TID:.*]]: !u32i):
// PARALLEL-EH:           cir.resume.flat %[[R_EXN]], %[[R_TID]]
// PARALLEL-EH:           omp.terminator
// PARALLEL-EH:         ^[[TERMINATE]]:
// PARALLEL-EH-NEXT:      %[[T_EXN:.*]], %{{.*}} = cir.eh.inflight_exception catch_all
// PARALLEL-EH-NEXT:      cir.call @__clang_call_terminate(%[[T_EXN]]) nothrow {noreturn}
// PARALLEL-EH-NEXT:      cir.unreachable
// PARALLEL-EH-NEXT:    }
// PARALLEL-EH-NEXT:    cir.return

// PARALLEL-OGCG-LABEL: define internal void @_Z25catch_by_copy_in_parallelv.omp_outlined(
// PARALLEL-OGCG-SAME:    personality ptr @__gxx_personality_v0
// PARALLEL-OGCG:         invoke void @_Z8mayThrowv()
// PARALLEL-OGCG:         landingpad { ptr, i32 }
// PARALLEL-OGCG-NEXT:            catch ptr @_ZTI11MyException
// PARALLEL-OGCG-NEXT:            catch ptr null
// PARALLEL-OGCG:         call ptr @__cxa_get_exception_ptr(ptr %{{.*}})
// PARALLEL-OGCG:         invoke void @_ZN11MyExceptionC1ERKS_(ptr {{.*}} %e, ptr {{.*}})
// PARALLEL-OGCG-NEXT:            to label %{{.*}} unwind label %[[TERMINATE:.*]]
// PARALLEL-OGCG:       [[TERMINATE]]:
// PARALLEL-OGCG-NEXT:    %[[T_LP:.*]] = landingpad { ptr, i32 }
// PARALLEL-OGCG-NEXT:            catch ptr null
// PARALLEL-OGCG-NEXT:    %[[T_EXN:.*]] = extractvalue { ptr, i32 } %[[T_LP]], 0
// PARALLEL-OGCG-NEXT:    call void @__clang_call_terminate(ptr %[[T_EXN]])
