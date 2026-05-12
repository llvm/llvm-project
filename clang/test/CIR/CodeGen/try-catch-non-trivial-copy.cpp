// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: cir-opt -cir-hoist-allocas -cir-flatten-cfg %t.cir -o %t.flat.cir
// RUN: FileCheck --input-file=%t.flat.cir %s -check-prefix=CIR-FLAT
// RUN: cir-opt -cir-hoist-allocas -cir-flatten-cfg -cir-eh-abi-lowering %t.cir -o %t.eh.cir
// RUN: FileCheck --input-file=%t.eh.cir %s -check-prefix=CIR-AFTER-EHABI
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

void mayThrow();

//===----------------------------------------------------------------------===//
// Case 1: Plain non-trivial copy constructor `T(const T &)`.
//===----------------------------------------------------------------------===//

struct MyException {
  MyException();
  MyException(const MyException &);
  ~MyException();
  int get();
};

int test_non_trivial_exception_copy() {
  int rv = 0;
  try {
    mayThrow();
  } catch (MyException e) {
    rv = e.get();
  }
  return rv;
}

// --- CIR (out of CodeGen, before any pre-lowering passes) ---
//
// The catch-binding step is `cir.construct_catch_param non_trivial_copy`,
// which references a CIRGen-synthesized helper thunk via `copy_fn`. The
// helper is keyed off the catch type's mangled typeinfo name, has
// `linkonce_odr` linkage, hidden visibility, and the
// `cir.eh.catch_copy_thunk` attribute so EHABI lowering can find or remove
// it later.

// CIR-LABEL: cir.func {{.*}} @_Z31test_non_trivial_exception_copyv()
// CIR:         %[[RETVAL:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR:         %[[RV:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["rv", init]
// CIR:         cir.scope {
// CIR:           %[[E:.*]] = cir.alloca !rec_MyException, !cir.ptr<!rec_MyException>, ["e"]
// CIR:           cir.try {
// CIR:             cir.call @_Z8mayThrowv() : () -> ()
// CIR:           } catch [type #cir.global_view<@_ZTI11MyException> : !cir.ptr<!u8i>] (%[[EH_TOKEN:.*]]: !cir.eh_token {{.*}}) {
// CIR:             cir.construct_catch_param non_trivial_copy %[[EH_TOKEN]] to %[[E]] using @__clang_cir_catch_copy__ZTS11MyException : !cir.ptr<!rec_MyException>
// CIR:             %[[CATCH_TOKEN:.*]], %[[EXN_PTR:.*]] = cir.begin_catch %[[EH_TOKEN]] : !cir.eh_token -> (!cir.catch_token, !cir.ptr<!void>)
// CIR:             cir.cleanup.scope {
// CIR:               cir.init_catch_param non_trivial_copy %[[EXN_PTR]] to %[[E]] : !cir.ptr<!void>, !cir.ptr<!rec_MyException>
// CIR:               cir.cleanup.scope {
// CIR:                 %[[GET:.*]] = cir.call @_ZN11MyException3getEv(%[[E]]) : (!cir.ptr<!rec_MyException> {{.*}}) -> (!s32i {{.*}})
// CIR:                 cir.store{{.*}} %[[GET]], %[[RV]]
// CIR:                 cir.yield
// CIR:               } cleanup all {
// CIR:                 cir.call @_ZN11MyExceptionD1Ev(%[[E]]) nothrow
// CIR:                 cir.yield
// CIR:               }
// CIR:               cir.yield
// CIR:             } cleanup all {
// CIR:               cir.end_catch %[[CATCH_TOKEN]] : !cir.catch_token
// CIR:               cir.yield
// CIR:             }
// CIR:           } unwind (%[[EH_TOKEN:.*]]: !cir.eh_token {{.*}}) {
// CIR:             cir.resume %[[EH_TOKEN]] : !cir.eh_token
// CIR:           }
// CIR:         }
// CIR:         cir.return

// CIR-LABEL: cir.func linkonce_odr hidden @__clang_cir_catch_copy__ZTS11MyException
// CIR-SAME:    attributes {cir.eh.catch_copy_thunk}
// CIR:         cir.call @_ZN11MyExceptionC1ERKS_(%{{.*}}, %{{.*}})
// CIR:         cir.return

// --- CIR-FLAT (after hoist-allocas + flatten-cfg) ---
//
// Regions are inlined into a flat CFG. The catch-binding op survives
// flattening unchanged; its `copy_fn` symbol reference is preserved.
//
// Pure-fallthrough blocks (just a `cir.br` to the next block) are skipped in
// the checks; we only verify branch terminators with non-trivial successors
// and the labels of blocks that carry real work.

// CIR-FLAT-LABEL: cir.func {{.*}} @_Z31test_non_trivial_exception_copyv()
// CIR-FLAT:         cir.try_call @_Z8mayThrowv() ^[[T1F_CONT:bb[0-9]+]], ^[[T1F_LPAD:bb[0-9]+]] : () -> ()

// On the unwind edge: initiate the in-flight exception and feed it to the
// dispatch block.
// CIR-FLAT:       ^[[T1F_LPAD]]:
// CIR-FLAT:         %[[T1F_TOK:.*]] = cir.eh.initiate : !cir.eh_token
// CIR-FLAT:         cir.br ^[[T1F_DISP:bb[0-9]+]](%[[T1F_TOK]] : !cir.eh_token)

// Dispatch matches the in-flight exception against `MyException` and falls
// through to the resume block on a miss.
// CIR-FLAT:       ^[[T1F_DISP]](%{{.*}}: !cir.eh_token):
// CIR-FLAT:         cir.eh.dispatch %{{.*}} : !cir.eh_token  [
// CIR-FLAT:           catch(#cir.global_view<@_ZTI11MyException> : !cir.ptr<!u8i>) : ^[[T1F_CATCH:bb[0-9]+]],
// CIR-FLAT:           unwind : ^[[T1F_RESUME:bb[0-9]+]]
// CIR-FLAT:         ]

// Catch handler: bind exception via the helper thunk, begin the catch,
// initialize the catch parameter, run the body.
// CIR-FLAT:       ^[[T1F_CATCH]](%{{.*}}: !cir.eh_token):
// CIR-FLAT:         cir.construct_catch_param non_trivial_copy %{{.*}} to %{{.*}} using @__clang_cir_catch_copy__ZTS11MyException : !cir.ptr<!rec_MyException>
// CIR-FLAT:         cir.begin_catch %{{.*}} : !cir.eh_token -> (!cir.catch_token, !cir.ptr<!void>)
// CIR-FLAT:         cir.init_catch_param non_trivial_copy %{{.*}} to %{{.*}} : !cir.ptr<!void>, !cir.ptr<!rec_MyException>
// CIR-FLAT:         cir.try_call @_ZN11MyException3getEv(%{{.*}}) ^[[T1F_GET_OK:bb[0-9]+]], ^[[T1F_GET_LPAD:bb[0-9]+]]

// Normal path of `e.get()`: store result and run `e`'s dtor.
// CIR-FLAT:       ^[[T1F_GET_OK]]:
// CIR-FLAT:         cir.store align(4) %{{.*}}
// CIR-FLAT:         cir.call @_ZN11MyExceptionD1Ev(%{{.*}}) nothrow

// Unwind path of `e.get()`: initiate cleanup, run `e`'s dtor, end cleanup.
// CIR-FLAT:       ^[[T1F_GET_LPAD]]:
// CIR-FLAT:         %[[T1F_CL_TOK:.*]] = cir.eh.initiate cleanup : !cir.eh_token
// CIR-FLAT:         cir.br ^[[T1F_CL_DTOR:bb[0-9]+]](%[[T1F_CL_TOK]] : !cir.eh_token)

// CIR-FLAT:       ^[[T1F_CL_DTOR]](%[[T1F_CL_TOK2:.*]]: !cir.eh_token):
// CIR-FLAT:         cir.begin_cleanup %[[T1F_CL_TOK2]] : !cir.eh_token -> !cir.cleanup_token
// CIR-FLAT:         cir.call @_ZN11MyExceptionD1Ev(%{{.*}}) nothrow
// CIR-FLAT:         cir.end_cleanup %{{.*}} : !cir.cleanup_token
// CIR-FLAT:         cir.br ^[[T1F_CL_END:bb[0-9]+]](%[[T1F_CL_TOK2]] : !cir.eh_token)

// Normal-path end_catch (after the catch body completed).
// CIR-FLAT:         cir.end_catch

// End of the catch's unwind path: end_catch then resume.
// CIR-FLAT:       ^[[T1F_CL_END]](%[[T1F_RES_TOK:.*]]: !cir.eh_token):
// CIR-FLAT:         cir.begin_cleanup
// CIR-FLAT:         cir.end_catch
// CIR-FLAT:         cir.end_cleanup
// CIR-FLAT:         cir.resume %[[T1F_RES_TOK]] : !cir.eh_token

// The outer dispatch unwind block (unmatched type).
// CIR-FLAT:       ^[[T1F_RESUME]](%[[T1F_OUT_TOK:.*]]: !cir.eh_token):
// CIR-FLAT:         cir.resume %[[T1F_OUT_TOK]] : !cir.eh_token

// Final return.
// CIR-FLAT:         cir.return %{{.*}} : !s32i

// CIR-AFTER-EHABI-LABEL: cir.func {{.*}} @_Z31test_non_trivial_exception_copyv()
// CIR-AFTER-EHABI:         cir.try_call @_Z8mayThrowv() ^[[T1E_CONT:bb[0-9]+]], ^[[T1E_LPAD:bb[0-9]+]]

// CIR-AFTER-EHABI:       ^[[T1E_LPAD]]:
// CIR-AFTER-EHABI:         %[[T1E_EXN:exception_ptr]], %[[T1E_TID:type_id]] = cir.eh.inflight_exception [@_ZTI11MyException]
// CIR-AFTER-EHABI:         cir.br ^{{bb[0-9]+}}(%[[T1E_EXN]], %[[T1E_TID]] : !cir.ptr<!void>, !u32i)

// At the dispatch: compare type_ids and branch to either the catch or resume.
// CIR-AFTER-EHABI:         %[[T1E_TYPEID:.*]] = cir.eh.typeid @_ZTI11MyException
// CIR-AFTER-EHABI:         %[[T1E_EQ:.*]] = cir.cmp eq %{{.*}}, %[[T1E_TYPEID]] : !u32i
// CIR-AFTER-EHABI:         cir.brcond %[[T1E_EQ]] ^[[T1E_CATCH:bb[0-9]+]](%{{.*}}, %{{.*}} : !cir.ptr<!void>, !u32i), ^[[T1E_RESUME:bb[0-9]+]](%{{.*}}, %{{.*}} : !cir.ptr<!void>, !u32i)

// In the catch block: get the adjusted exception pointer, then invoke the
// inlined copy ctor directly. The normal-edge target is a pure-fallthrough
// block that we don't pin; the unwind-edge target is the terminate handler.
// CIR-AFTER-EHABI:       ^[[T1E_CATCH]](%[[T1E_RAW:.*]]: !cir.ptr<!void>, %{{.*}}: !u32i):
// CIR-AFTER-EHABI:         %[[T1E_ADJ:.*]] = cir.call @__cxa_get_exception_ptr(%[[T1E_RAW]]) nothrow : (!cir.ptr<!void>) -> !cir.ptr<!u8i>
// CIR-AFTER-EHABI:         %[[T1E_ADJT:.*]] = cir.cast bitcast %[[T1E_ADJ]] : !cir.ptr<!u8i> -> !cir.ptr<!rec_MyException>
// CIR-AFTER-EHABI:         cir.try_call @_ZN11MyExceptionC1ERKS_(%{{.*}}, %[[T1E_ADJT]]) ^{{bb[0-9]+}}, ^[[T1E_TERM:bb[0-9]+]]

// On the inlined copy's normal-edge fallthrough chain: __cxa_begin_catch and
// run the catch body.
// CIR-AFTER-EHABI:         cir.call @__cxa_begin_catch(%[[T1E_RAW]]) : (!cir.ptr<!void>) -> !cir.ptr<!u8i>
// CIR-AFTER-EHABI:         cir.try_call @_ZN11MyException3getEv(%{{.*}}) ^[[T1E_GET_OK:bb[0-9]+]], ^[[T1E_GET_LPAD:bb[0-9]+]]

// Normal path: store get's result, dtor, end_catch, fall through.
// CIR-AFTER-EHABI:       ^[[T1E_GET_OK]]:
// CIR-AFTER-EHABI:         cir.store align(4) %{{.*}}
// CIR-AFTER-EHABI:         cir.call @_ZN11MyExceptionD1Ev(%{{.*}}) nothrow

// Unwind path: inflight_exception cleanup, dtor, end_catch, resume.
// CIR-AFTER-EHABI:       ^[[T1E_GET_LPAD]]:
// CIR-AFTER-EHABI:         %{{.*}}, %{{.*}} = cir.eh.inflight_exception cleanup
// CIR-AFTER-EHABI:         cir.call @_ZN11MyExceptionD1Ev(%{{.*}}) nothrow
// CIR-AFTER-EHABI:         cir.call @__cxa_end_catch() : () -> ()
// CIR-AFTER-EHABI:         cir.call @__cxa_end_catch() : () -> ()
// CIR-AFTER-EHABI:         cir.resume.flat %{{.*}}

// Outer dispatch's resume.
// CIR-AFTER-EHABI:       ^[[T1E_RESUME]](%{{.*}}: !cir.ptr<!void>, %{{.*}}: !u32i):
// CIR-AFTER-EHABI:         cir.resume.flat %{{.*}}, %{{.*}}

// CIR-AFTER-EHABI:         cir.return %{{.*}} : !s32i

// Terminate landing pad.
// CIR-AFTER-EHABI:       ^[[T1E_TERM]]:
// CIR-AFTER-EHABI:         %{{.*}}, %{{.*}} = cir.eh.inflight_exception catch_all
// CIR-AFTER-EHABI:         cir.call @__clang_call_terminate(%{{.*}}) nothrow {noreturn} : (!cir.ptr<!void>) -> ()
// CIR-AFTER-EHABI:         cir.unreachable


// The catch-copy thunk is fully inlined and removed by the post-lowering
// sweep, so it is no longer present at this stage.
// CIR-AFTER-EHABI-NOT: @__clang_cir_catch_copy__ZTS11MyException

// --- LLVM (CIR -> LLVM IR via the full pipeline) ---

// LLVM-LABEL: define dso_local noundef i32 @_Z31test_non_trivial_exception_copyv() #{{[0-9]+}} personality ptr @__gxx_personality_v0
// LLVM:         invoke void @_Z8mayThrowv()
// LLVM:                 to label %[[T1L_CONT:.*]] unwind label %[[T1L_LPAD:.*]]

// LLVM:       [[T1L_LPAD]]:
// LLVM:         landingpad { ptr, i32 }
// LLVM:                 catch ptr @_ZTI11MyException

// Type-id dispatch.
// LLVM:         %[[T1L_TYPEID:.*]] = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTI11MyException)
// LLVM:         %[[T1L_MATCH:.*]] = icmp eq i32 %{{.*}}, %[[T1L_TYPEID]]
// LLVM:         br i1 %[[T1L_MATCH]], label %[[T1L_CATCH:.*]], label %[[T1L_RESUME:.*]]

// Catch handler: __cxa_get_exception_ptr followed by the inlined copy ctor
// invoke (the catch-copy thunk has been inlined away, so the ctor is called
// directly with a terminate landing pad). The normal-edge target is a
// pure-fallthrough block that we don't pin.
// LLVM:       [[T1L_CATCH]]:
// LLVM:         %[[T1L_ADJ:.*]] = call ptr @__cxa_get_exception_ptr(
// LLVM:         invoke void @_ZN11MyExceptionC1ERKS_(ptr {{.*}}, ptr{{.*}} %[[T1L_ADJ]])
// LLVM:                 to label %{{.*}} unwind label %[[T1L_TERM:.*]]

// On the inlined copy's normal-edge fallthrough chain: __cxa_begin_catch.
// LLVM:         call ptr @__cxa_begin_catch(
// LLVM:         invoke noundef i32 @_ZN11MyException3getEv(
// LLVM:                 to label %[[T1L_GET_OK:.*]] unwind label %[[T1L_GET_LPAD:.*]]

// LLVM:       [[T1L_GET_OK]]:
// LLVM:         store i32 %{{.*}}
// LLVM:         call void @_ZN11MyExceptionD1Ev(

// LLVM:       [[T1L_GET_LPAD]]:
// LLVM:         landingpad { ptr, i32 }
// LLVM:                 cleanup
// LLVM:         call void @_ZN11MyExceptionD1Ev(
// LLVM:         call void @__cxa_end_catch()
// LLVM:         call void @__cxa_end_catch()
// LLVM:         resume { ptr, i32 }

// LLVM:       [[T1L_RESUME]]:
// LLVM:         resume { ptr, i32 }
// LLVM:         ret i32

// LLVM:       [[T1L_TERM]]:
// LLVM:         landingpad { ptr, i32 }
// LLVM:                 catch ptr null
// LLVM:         call void @__clang_call_terminate(
// LLVM:         unreachable

// The catch-copy thunk is inlined away during EHABI lowering, so it should
// not appear in the LLVM IR.
// LLVM-NOT: @__clang_cir_catch_copy__ZTS11MyException

// --- OGCG (classic CodeGen reference) ---
//
// Classic CodeGen does not synthesize a catch-init thunk; it inlines the
// `__cxa_get_exception_ptr` + copy ctor invocation at the catch site, with a
// terminate landing pad on the copy's unwind edge.

// OGCG-LABEL: define {{.*}} i32 @_Z31test_non_trivial_exception_copyv()
// OGCG:       entry:
// OGCG:         %[[O1_RV:.*]] = alloca i32
// OGCG:         %[[O1_E:.*]] = alloca %struct.MyException
// OGCG:         store i32 0, ptr %[[O1_RV]]
// OGCG:         invoke void @_Z8mayThrowv()
// OGCG:                 to label %[[O1_CONT:.*]] unwind label %[[O1_LPAD:.*]]

// OGCG:       [[O1_LPAD]]:
// OGCG:         landingpad { ptr, i32 }
// OGCG:                 catch ptr @_ZTI11MyException

// OGCG:       catch.dispatch:
// OGCG:         %[[O1_TYPEID:.*]] = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTI11MyException)
// OGCG:         %[[O1_MATCH:.*]] = icmp eq i32 %{{.*}}, %[[O1_TYPEID]]
// OGCG:         br i1 %[[O1_MATCH]], label %[[O1_CATCH:.*]], label %[[EH_RESUME:.*]]

// OGCG:       [[O1_CATCH]]:
// OGCG:         %[[O1_ADJ:.*]] = call ptr @__cxa_get_exception_ptr(ptr %{{.*}})
// OGCG:         invoke void @_ZN11MyExceptionC1ERKS_(ptr noundef nonnull align 1 dereferenceable(1) %[[O1_E]], ptr noundef nonnull align 1 dereferenceable(1) %[[O1_ADJ]])
// OGCG:                 to label %[[O1_INVOKE_CONT1:.*]] unwind label %[[O1_TERM:.*]]

// OGCG:       [[O1_INVOKE_CONT1]]:
// OGCG:         call ptr @__cxa_begin_catch(
// OGCG:         invoke noundef i32 @_ZN11MyException3getEv(ptr noundef nonnull align 1 dereferenceable(1) %[[O1_E]])
// OGCG:                 to label %[[O1_INVOKE_CONT3:.*]] unwind label %[[O1_LPAD2:.*]]

// OGCG:       [[O1_INVOKE_CONT3]]:
// OGCG:         store i32 %{{.*}}, ptr %[[O1_RV]]
// OGCG:         call void @_ZN11MyExceptionD1Ev(ptr noundef nonnull align 1 dereferenceable(1) %[[O1_E]])
// OGCG:         call void @__cxa_end_catch()
// OGCG:         br label %[[O1_TRY_CONT:.*]]

// OGCG:       [[O1_TRY_CONT]]:
// OGCG:         load i32, ptr %[[O1_RV]]
// OGCG:         ret i32

// OGCG:       [[O1_LPAD2]]:
// OGCG:         landingpad { ptr, i32 }
// OGCG:                 cleanup
// OGCG:         call void @_ZN11MyExceptionD1Ev(ptr noundef nonnull align 1 dereferenceable(1) %[[O1_E]])
// OGCG:         invoke void @__cxa_end_catch()
// OGCG:                 to label %{{.*}} unwind label %[[O1_TERM]]

// OGCG:       [[EH_RESUME]]:
// OGCG:         resume { ptr, i32 }

// OGCG:       [[O1_TERM]]:
// OGCG:         landingpad { ptr, i32 }
// OGCG:                 catch ptr null
// OGCG:         call void @__clang_call_terminate(
// OGCG:         unreachable

//===----------------------------------------------------------------------===//
// Case 2: Copy constructor with extra (default) arguments.
//
// `WithDefault(const WithDefault &, int = 42)` is a valid copy constructor
// per [class.copy.ctor]/1 since the only non-`const T &` parameter has a
// default argument. emitAggExpr lets the CXXDefaultArgExpr fill-in flow
// through naturally; the thunk body materializes the `42` constant before
// calling the underlying ctor.
//===----------------------------------------------------------------------===//

struct WithDefault {
  WithDefault();
  WithDefault(const WithDefault &, int = 42);
  ~WithDefault();
};

int test_copy_ctor_extra_args() {
  try {
    mayThrow();
  } catch (WithDefault w) {
    return 0;
  }
  return -1;
}

// --- CIR ---

// CIR-LABEL: cir.func {{.*}} @_Z25test_copy_ctor_extra_argsv()
// CIR:         cir.construct_catch_param non_trivial_copy %{{.*}} to %{{.*}} using @__clang_cir_catch_copy__ZTS11WithDefault : !cir.ptr<!rec_WithDefault>

// CIR-LABEL: cir.func linkonce_odr hidden @__clang_cir_catch_copy__ZTS11WithDefault(
// CIR-SAME:    attributes {cir.eh.catch_copy_thunk}
// CIR:         %[[FORTYTWO:.*]] = cir.const #cir.int<42> : !s32i
// CIR:         cir.call @_ZN11WithDefaultC1ERKS_i({{.*}}, {{.*}}, %[[FORTYTWO]])
// CIR:         cir.return

// --- CIR-FLAT ---
//
// Because the catch handler does `return 0`, the function uses a
// `__cleanup_dest_slot` to dispatch through the catch end_catch and into
// either the inner-`return 0` or the outer-`return -1`. We don't pin the
// exact switch shape here, but we do verify the EH-relevant control flow.

// CIR-FLAT-LABEL: cir.func {{.*}} @_Z25test_copy_ctor_extra_argsv()
// CIR-FLAT:         %{{.*}} = cir.alloca !s32i, !cir.ptr<!s32i>, ["__cleanup_dest_slot", cleanup_dest_slot]
// CIR-FLAT:         cir.try_call @_Z8mayThrowv() ^[[T2F_CONT:bb[0-9]+]], ^[[T2F_LPAD:bb[0-9]+]]

// CIR-FLAT:       ^[[T2F_LPAD]]:
// CIR-FLAT:         %[[T2F_TOK:.*]] = cir.eh.initiate : !cir.eh_token
// CIR-FLAT:         cir.br ^[[T2F_DISP:bb[0-9]+]](%[[T2F_TOK]] : !cir.eh_token)

// CIR-FLAT:       ^[[T2F_DISP]](%{{.*}}: !cir.eh_token):
// CIR-FLAT:         cir.eh.dispatch %{{.*}} : !cir.eh_token  [
// CIR-FLAT:           catch(#cir.global_view<@_ZTI11WithDefault> : !cir.ptr<!u8i>) : ^[[T2F_CATCH:bb[0-9]+]],
// CIR-FLAT:           unwind : ^[[T2F_RESUME:bb[0-9]+]]
// CIR-FLAT:         ]

// CIR-FLAT:       ^[[T2F_CATCH]](%{{.*}}: !cir.eh_token):
// CIR-FLAT:         cir.construct_catch_param non_trivial_copy %{{.*}} to %{{.*}} using @__clang_cir_catch_copy__ZTS11WithDefault : !cir.ptr<!rec_WithDefault>
// CIR-FLAT:         cir.begin_catch %{{.*}} : !cir.eh_token -> (!cir.catch_token, !cir.ptr<!void>)
// CIR-FLAT:         cir.init_catch_param non_trivial_copy %{{.*}} to %{{.*}}

// Catch body sets retval=0 and cleanup_dest=1.
// CIR-FLAT:         cir.store %{{.*}}
// CIR-FLAT:         cir.store %{{.*}}

// `w`'s destructor.
// CIR-FLAT:         cir.call @_ZN11WithDefaultD1Ev(%{{.*}}) nothrow

// Cleanup-dest dispatch and end_catch.
// CIR-FLAT:         cir.switch.flat
// CIR-FLAT:         cir.end_catch

// Return-or-fallthrough switch on cleanup_dest_slot.
// CIR-FLAT:         cir.switch.flat
// CIR-FLAT:         cir.load %{{.*}} : !cir.ptr<!s32i>, !s32i
// CIR-FLAT:         cir.return %{{.*}} : !s32i

// CIR-FLAT:       ^[[T2F_RESUME]](%[[T2F_OUT_TOK:.*]]: !cir.eh_token):
// CIR-FLAT:         cir.resume %[[T2F_OUT_TOK]] : !cir.eh_token

// Tail of the function (when the try doesn't catch):
// CIR-FLAT:         cir.return %{{.*}} : !s32i

// --- CIR-AFTER-EHABI ---

// CIR-AFTER-EHABI-LABEL: cir.func {{.*}} @_Z25test_copy_ctor_extra_argsv()
// CIR-AFTER-EHABI:         cir.try_call @_Z8mayThrowv() ^[[T2E_CONT:bb[0-9]+]], ^[[T2E_LPAD:bb[0-9]+]]

// CIR-AFTER-EHABI:       ^[[T2E_LPAD]]:
// CIR-AFTER-EHABI:         %{{.*}}, %{{.*}} = cir.eh.inflight_exception [@_ZTI11WithDefault]

// CIR-AFTER-EHABI:         %[[T2E_TYPEID:.*]] = cir.eh.typeid @_ZTI11WithDefault
// CIR-AFTER-EHABI:         %[[T2E_EQ:.*]] = cir.cmp eq %{{.*}}, %[[T2E_TYPEID]]
// CIR-AFTER-EHABI:         cir.brcond %[[T2E_EQ]] ^[[T2E_CATCH:bb[0-9]+]](%{{.*}}, %{{.*}} : !cir.ptr<!void>, !u32i), ^[[T2E_RESUME:bb[0-9]+]](%{{.*}}, %{{.*}} : !cir.ptr<!void>, !u32i)

// In the catch block: get the adjusted exception pointer, materialize the
// `42` default argument, and invoke the inlined copy ctor directly. The
// thunk has been inlined into the catch handler, so the constant comes from
// the cloned thunk body. The normal-edge target is a pure-fallthrough block
// that we don't pin; the unwind-edge target is the terminate handler.
// CIR-AFTER-EHABI:       ^[[T2E_CATCH]](%[[T2E_RAW:.*]]: !cir.ptr<!void>, %{{.*}}: !u32i):
// CIR-AFTER-EHABI:         %[[T2E_ADJ:.*]] = cir.call @__cxa_get_exception_ptr(%[[T2E_RAW]]) nothrow
// CIR-AFTER-EHABI:         %[[T2E_ADJT:.*]] = cir.cast bitcast %[[T2E_ADJ]] : !cir.ptr<!u8i> -> !cir.ptr<!rec_WithDefault>
// CIR-AFTER-EHABI:         %[[T2E_FORTYTWO:.*]] = cir.const #cir.int<42> : !s32i
// CIR-AFTER-EHABI:         cir.try_call @_ZN11WithDefaultC1ERKS_i(%{{.*}}, %[[T2E_ADJT]], %[[T2E_FORTYTWO]]) ^{{bb[0-9]+}}, ^[[T2E_TERM:bb[0-9]+]]

// On the inlined copy's normal-edge fallthrough chain: __cxa_begin_catch.
// CIR-AFTER-EHABI:         cir.call @__cxa_begin_catch(%[[T2E_RAW]])

// Catch body and dispatch out via cleanup_dest_slot.
// CIR-AFTER-EHABI:         cir.call @_ZN11WithDefaultD1Ev(%{{.*}}) nothrow
// CIR-AFTER-EHABI:         cir.switch.flat
// CIR-AFTER-EHABI:         cir.call @__cxa_end_catch()
// CIR-AFTER-EHABI:         cir.switch.flat
// CIR-AFTER-EHABI:         cir.return %{{.*}} : !s32i

// CIR-AFTER-EHABI:       ^[[T2E_RESUME]](%{{.*}}: !cir.ptr<!void>, %{{.*}}: !u32i):
// CIR-AFTER-EHABI:         cir.resume.flat %{{.*}}, %{{.*}}

// CIR-AFTER-EHABI:         cir.return %{{.*}} : !s32i

// Terminate landing pad.
// CIR-AFTER-EHABI:       ^[[T2E_TERM]]:
// CIR-AFTER-EHABI:         cir.eh.inflight_exception catch_all
// CIR-AFTER-EHABI:         cir.call @__clang_call_terminate(

// The catch-copy thunk is fully inlined and removed.
// CIR-AFTER-EHABI-NOT: @__clang_cir_catch_copy__ZTS11WithDefault

// --- LLVM ---

// LLVM-LABEL: define dso_local noundef i32 @_Z25test_copy_ctor_extra_argsv()
// LLVM:         invoke void @_Z8mayThrowv()
// LLVM:                 to label %[[T2L_CONT:.*]] unwind label %[[T2L_LPAD:.*]]

// LLVM:       [[T2L_LPAD]]:
// LLVM:         landingpad { ptr, i32 }
// LLVM:                 catch ptr @_ZTI11WithDefault

// LLVM:         %[[T2L_TYPEID:.*]] = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTI11WithDefault)
// LLVM:         icmp eq i32 %{{.*}}, %[[T2L_TYPEID]]
// LLVM:         br i1 %{{.*}}, label %[[T2L_CATCH:.*]], label %[[T2L_RESUME:.*]]

// LLVM:       [[T2L_CATCH]]:
// LLVM:         %[[T2L_ADJ:.*]] = call ptr @__cxa_get_exception_ptr(
// LLVM:         invoke void @_ZN11WithDefaultC1ERKS_i(ptr {{.*}}, ptr {{.*}} %[[T2L_ADJ]], i32 {{.*}} 42)
// LLVM:                 to label %{{.*}} unwind label %[[T2L_TERM:.*]]

// On the inlined copy's normal-edge fallthrough chain: __cxa_begin_catch.
// LLVM:         call ptr @__cxa_begin_catch(

// LLVM:         call void @_ZN11WithDefaultD1Ev(
// LLVM:         switch i32 %{{.*}}
// LLVM:         call void @__cxa_end_catch()
// LLVM:         switch i32 %{{.*}}
// LLVM:         ret i32 %{{.*}}

// LLVM:       [[T2L_RESUME]]:
// LLVM:         resume { ptr, i32 }
// LLVM:         ret i32

// LLVM:       [[T2L_TERM]]:
// LLVM:         landingpad { ptr, i32 }
// LLVM:                 catch ptr null
// LLVM:         call void @__clang_call_terminate(
// LLVM:         unreachable

// The catch-copy thunk is inlined away; no separate function definition.
// LLVM-NOT: @__clang_cir_catch_copy__ZTS11WithDefault

// --- OGCG ---

// OGCG-LABEL: define {{.*}} i32 @_Z25test_copy_ctor_extra_argsv()
// OGCG:       entry:
// OGCG:         invoke void @_Z8mayThrowv()
// OGCG:                 to label %[[O2_CONT:.*]] unwind label %[[O2_LPAD:.*]]

// OGCG:       [[O2_LPAD]]:
// OGCG:         landingpad { ptr, i32 }
// OGCG:                 catch ptr @_ZTI11WithDefault

// OGCG:       catch.dispatch:
// OGCG:         %[[O2_TYPEID:.*]] = call i32 @llvm.eh.typeid.for.p0(ptr @_ZTI11WithDefault)
// OGCG:         icmp eq i32 %{{.*}}, %[[O2_TYPEID]]
// OGCG:         br i1 %{{.*}}, label %[[O2_CATCH:.*]], label %[[EH_RESUME:.*]]

// OGCG:       [[O2_CATCH]]:
// OGCG:         %[[O2_ADJ:.*]] = call ptr @__cxa_get_exception_ptr(ptr %{{.*}})
// OGCG:         invoke void @_ZN11WithDefaultC1ERKS_i(ptr {{.*}} %{{.*}}, ptr {{.*}} %[[O2_ADJ]], i32 {{.*}} 42)
// OGCG:                 to label %[[O2_INVOKE_CONT1:.*]] unwind label %[[O2_TERM:.*]]

// OGCG:       [[O2_INVOKE_CONT1]]:
// OGCG:         call ptr @__cxa_begin_catch(
// OGCG:         store i32 0, ptr %{{.*}}
// OGCG:         call void @_ZN11WithDefaultD1Ev(
// OGCG:         call void @__cxa_end_catch()
// OGCG:         br label %[[RETURN:.*]]

// OGCG:       [[TRY_CONT:.*]]:
// OGCG:         store i32 -1, ptr %{{.*}}
// OGCG:         br label %[[RETURN:.*]]

// OGCG:       [[RETURN]]:
// OGCG:         load i32, ptr %{{.*}}
// OGCG:         ret i32

// OGCG:       eh.resume:
// OGCG:         resume { ptr, i32 }

// OGCG:       [[O2_TERM]]:
// OGCG:         landingpad { ptr, i32 }
// OGCG:                 catch ptr null
// OGCG:         call void @__clang_call_terminate(

//===----------------------------------------------------------------------===//
// Module-level checks (runtime helpers materialized by EHABI lowering).
//
// These come last so they aren't partitioned by any per-function LABEL.
//===----------------------------------------------------------------------===//

// CIR-AFTER-EHABI: cir.func private @__cxa_begin_catch(
// CIR-AFTER-EHABI: cir.func private @__cxa_end_catch()
// CIR-AFTER-EHABI: cir.func private @__cxa_get_exception_ptr(
// CIR-AFTER-EHABI: cir.func linkonce_odr hidden @__clang_call_terminate(
// CIR-AFTER-EHABI: cir.func private @_ZSt9terminatev()
