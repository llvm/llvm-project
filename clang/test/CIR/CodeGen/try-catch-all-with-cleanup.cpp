// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: cir-opt %t.cir -cir-flatten-cfg -o %t-flat.cir
// RUN: FileCheck --input-file=%t-flat.cir %s -check-prefix=CIR-FLAT
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

void mayThrow();

struct S {
  S();
  ~S();
};

void test_catch_all_with_cleanup() {
  try {
    S s;
    mayThrow();
  } catch (...) {
  }
}

// CIR-LABEL: cir.func {{.*}} @_Z27test_catch_all_with_cleanupv()
// CIR:   cir.scope {
// CIR:     %[[S:.*]] = cir.alloca !rec_S, !cir.ptr<!rec_S>, ["s", init]
// CIR:     cir.try {
// CIR:       cir.call @_ZN1SC1Ev(%[[S]])
// CIR:       cir.cleanup.scope {
// CIR:         cir.call @_Z8mayThrowv()
// CIR:         cir.yield
// CIR:       } cleanup all {
// CIR:         cir.call @_ZN1SD1Ev(%[[S]]) nothrow
// CIR:         cir.yield
// CIR:       }
// CIR:       cir.yield
// CIR:     } catch all (%{{.*}}: !cir.eh_token {{.*}}) {
// CIR:       %{{.*}}, %{{.*}} = cir.begin_catch
// CIR:       cir.cleanup.scope {
// CIR:         cir.yield
// CIR:       } cleanup all {
// CIR:         cir.end_catch
// CIR:         cir.yield
// CIR:       }
// CIR:       cir.yield
// CIR:     }
// CIR:   }

// CIR-FLAT-LABEL: cir.func {{.*}} @_Z27test_catch_all_with_cleanupv()
//
// CIR-FLAT:         %[[S:.*]] = cir.alloca !rec_S
//
// Ctor may throw; unwinds directly to the dispatch (no cleanup needed yet).
// CIR-FLAT:         cir.try_call @_ZN1SC1Ev(%[[S]]) ^[[AFTER_CTOR:bb[0-9]+]], ^[[CTOR_UNWIND:bb[0-9]+]]
//
// After the ctor, enter the cleanup scope where mayThrow may throw.
// CIR-FLAT:       ^[[AFTER_CTOR]]:
// CIR-FLAT:         cir.try_call @_Z8mayThrowv() ^[[NORMAL:bb[0-9]+]], ^[[INNER_UNWIND:bb[0-9]+]]
//
// Normal path: destroy s and exit the try.
// CIR-FLAT:       ^[[NORMAL]]:
// CIR-FLAT:         cir.call @_ZN1SD1Ev(%[[S]]) nothrow
// CIR-FLAT:         cir.br ^{{.*}}
//
// Inner unwind: NO cleanup flag — catch-all catches everything.
// CIR-FLAT:       ^[[INNER_UNWIND]]:
// CIR-FLAT:         %[[INNER_ET:.*]] = cir.eh.initiate : !cir.eh_token
// CIR-FLAT:         cir.br ^[[EH_CLEANUP:bb[0-9]+]](%[[INNER_ET]] : !cir.eh_token)
//
// EH cleanup: destroy s on the exception path, then go to dispatch.
// CIR-FLAT:       ^[[EH_CLEANUP]](%[[EH_ET:.*]]: !cir.eh_token):
// CIR-FLAT:         %[[CT:.*]] = cir.begin_cleanup %[[EH_ET]]
// CIR-FLAT:         cir.call @_ZN1SD1Ev(%[[S]]) nothrow
// CIR-FLAT:         cir.end_cleanup %[[CT]]
// CIR-FLAT:         cir.br ^[[DISPATCH:bb[0-9]+]](%[[EH_ET]] : !cir.eh_token)
//
// Ctor unwind: NO cleanup flag — goes directly to dispatch.
// CIR-FLAT:       ^[[CTOR_UNWIND]]:
// CIR-FLAT:         %[[CTOR_ET:.*]] = cir.eh.initiate : !cir.eh_token
// CIR-FLAT:         cir.br ^[[DISPATCH]](%[[CTOR_ET]] : !cir.eh_token)
//
// Dispatch: catch-all handler.
// CIR-FLAT:       ^[[DISPATCH]](%[[DISP_ET:.*]]: !cir.eh_token):
// CIR-FLAT:         cir.eh.dispatch %[[DISP_ET]] : !cir.eh_token [
// CIR-FLAT:           catch_all : ^[[CATCH_ALL:bb[0-9]+]]
// CIR-FLAT:         ]
//
// Catch handler.
// CIR-FLAT:       ^[[CATCH_ALL]](%[[CA_ET:.*]]: !cir.eh_token):
// CIR-FLAT:         %{{.*}}, %{{.*}} = cir.begin_catch %[[CA_ET]]
// CIR-FLAT:         cir.end_catch
// CIR-FLAT:         cir.return

// Both landing pads must use "catch ptr null" (not "cleanup") so that the
// personality function recognises a catch-all handler during the search phase.

// LLVM-LABEL: define {{.*}} void @_Z27test_catch_all_with_cleanupv()
// LLVM-SAME:    personality ptr @__gxx_personality_v0
// LLVM:         %[[S:.*]] = alloca %struct.S
// LLVM:         invoke void @_ZN1SC1Ev({{.*}}%[[S]])
// LLVM:                 to label %[[AFTER_CTOR:.*]] unwind label %[[CTOR_LP:.*]]
// LLVM:       [[AFTER_CTOR]]:
// LLVM:         invoke void @_Z8mayThrowv()
// LLVM:                 to label %[[NORMAL:.*]] unwind label %[[INNER_LP:.*]]
// LLVM:       [[NORMAL]]:
// LLVM:         call void @_ZN1SD1Ev({{.*}}%[[S]])
// LLVM:       [[INNER_LP]]:
// LLVM:         landingpad { ptr, i32 }
// LLVM:                 catch ptr null
// LLVM:         call void @_ZN1SD1Ev({{.*}}%[[S]])
// LLVM:       [[CTOR_LP]]:
// LLVM:         landingpad { ptr, i32 }
// LLVM:                 catch ptr null
// LLVM:         call ptr @__cxa_begin_catch
// LLVM:         call void @__cxa_end_catch()
// LLVM:         ret void

// OGCG-LABEL: define {{.*}} void @_Z27test_catch_all_with_cleanupv()
// OGCG-SAME:    personality ptr @__gxx_personality_v0
// OGCG:         %[[S:.*]] = alloca %struct.S
// OGCG:         invoke void @_ZN1SC1Ev({{.*}}%[[S]])
// OGCG:                 to label %[[AFTER_CTOR:.*]] unwind label %[[CTOR_LP:.*]]
// OGCG:       [[AFTER_CTOR]]:
// OGCG:         invoke void @_Z8mayThrowv()
// OGCG:                 to label %[[NORMAL:.*]] unwind label %[[INNER_LP:.*]]
// OGCG:       [[NORMAL]]:
// OGCG:         call void @_ZN1SD1Ev({{.*}}%[[S]])
// OGCG:       [[CTOR_LP]]:
// OGCG:         landingpad { ptr, i32 }
// OGCG:                 catch ptr null
// OGCG:       [[INNER_LP]]:
// OGCG:         landingpad { ptr, i32 }
// OGCG:                 catch ptr null
// OGCG:         call void @_ZN1SD1Ev({{.*}}%[[S]])
// OGCG:         call ptr @__cxa_begin_catch
// OGCG:         call void @__cxa_end_catch()
// OGCG:         ret void

void test_catch_all_and_specific_with_cleanup() {
  try {
    S s;
    mayThrow();
  } catch (int e) {
  } catch (...) {
  }
}

// CIR-LABEL: cir.func {{.*}} @_Z40test_catch_all_and_specific_with_cleanupv()
// CIR:   cir.scope {
// CIR:     %[[S:.*]] = cir.alloca !rec_S, !cir.ptr<!rec_S>, ["s", init]
// CIR:     %[[E:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["e"]
// CIR:     cir.try {
// CIR:       cir.call @_ZN1SC1Ev(%[[S]])
// CIR:       cir.cleanup.scope {
// CIR:         cir.call @_Z8mayThrowv()
// CIR:         cir.yield
// CIR:       } cleanup all {
// CIR:         cir.call @_ZN1SD1Ev(%[[S]]) nothrow
// CIR:         cir.yield
// CIR:       }
// CIR:       cir.yield
// CIR:     } catch [type #cir.global_view<@_ZTIi> : !cir.ptr<!u8i>] (%{{.*}}: !cir.eh_token {{.*}}) {
// CIR:       %{{.*}}, %[[EXN:.*]] = cir.begin_catch %{{.*}} : !cir.eh_token -> (!cir.catch_token, !cir.ptr<!s32i>)
// CIR:       cir.cleanup.scope {
// CIR:         cir.load{{.*}} %[[EXN]] : !cir.ptr<!s32i>, !s32i
// CIR:         cir.store{{.*}} %{{.*}}, %[[E]] : !s32i, !cir.ptr<!s32i>
// CIR:         cir.yield
// CIR:       } cleanup all {
// CIR:         cir.end_catch %{{.*}} : !cir.catch_token
// CIR:         cir.yield
// CIR:       }
// CIR:       cir.yield
// CIR:     } catch all (%{{.*}}: !cir.eh_token {{.*}}) {
// CIR:       %{{.*}}, %{{.*}} = cir.begin_catch %{{.*}} : !cir.eh_token -> (!cir.catch_token, !cir.ptr<!void>)
// CIR:       cir.cleanup.scope {
// CIR:         cir.yield
// CIR:       } cleanup all {
// CIR:         cir.end_catch %{{.*}} : !cir.catch_token
// CIR:         cir.yield
// CIR:       }
// CIR:       cir.yield
// CIR:     }
// CIR:   }

// CIR-FLAT-LABEL: cir.func {{.*}} @_Z40test_catch_all_and_specific_with_cleanupv()
//
// CIR-FLAT:         %[[S:.*]] = cir.alloca !rec_S
// CIR-FLAT:         %[[E:.*]] = cir.alloca !s32i
//
// Ctor may throw; unwinds directly to the dispatch (no cleanup needed yet).
// CIR-FLAT:         cir.try_call @_ZN1SC1Ev(%[[S]]) ^[[AFTER_CTOR:bb[0-9]+]], ^[[CTOR_UNWIND:bb[0-9]+]]
//
// After the ctor, enter the cleanup scope where mayThrow may throw.
// CIR-FLAT:       ^[[AFTER_CTOR]]:
// CIR-FLAT:         cir.try_call @_Z8mayThrowv() ^[[NORMAL:bb[0-9]+]], ^[[INNER_UNWIND:bb[0-9]+]]
//
// Normal path: destroy s and exit the try.
// CIR-FLAT:       ^[[NORMAL]]:
// CIR-FLAT:         cir.call @_ZN1SD1Ev(%[[S]]) nothrow
// CIR-FLAT:         cir.br ^{{.*}}
//
// Inner unwind: run cleanup (dtor of s), then dispatch int vs catch-all.
// CIR-FLAT:       ^[[INNER_UNWIND]]:
// CIR-FLAT:         %[[INNER_ET:.*]] = cir.eh.initiate : !cir.eh_token
// CIR-FLAT:         cir.br ^[[EH_CLEANUP:bb[0-9]+]](%[[INNER_ET]] : !cir.eh_token)
//
// EH cleanup: destroy s on the exception path, then go to dispatch.
// CIR-FLAT:       ^[[EH_CLEANUP]](%[[EH_ET:.*]]: !cir.eh_token):
// CIR-FLAT:         %[[CT:.*]] = cir.begin_cleanup %[[EH_ET]]
// CIR-FLAT:         cir.call @_ZN1SD1Ev(%[[S]]) nothrow
// CIR-FLAT:         cir.end_cleanup %[[CT]]
// CIR-FLAT:         cir.br ^[[DISPATCH:bb[0-9]+]](%[[EH_ET]] : !cir.eh_token)
//
// Ctor unwind: NO cleanup flag — goes directly to dispatch.
// CIR-FLAT:       ^[[CTOR_UNWIND]]:
// CIR-FLAT:         %[[CTOR_ET:.*]] = cir.eh.initiate : !cir.eh_token
// CIR-FLAT:         cir.br ^[[DISPATCH]](%[[CTOR_ET]] : !cir.eh_token)
//
// Dispatch: typed catch (int) first, then catch-all.
// CIR-FLAT:       ^[[DISPATCH]](%[[DISP_ET:.*]]: !cir.eh_token):
// CIR-FLAT:         cir.eh.dispatch %[[DISP_ET]] : !cir.eh_token [
// CIR-FLAT:           catch(#cir.global_view<@_ZTIi> : !cir.ptr<!u8i>) : ^[[CATCH_INT:bb[0-9]+]],
// CIR-FLAT:           catch_all : ^[[CATCH_ALL:bb[0-9]+]]
// CIR-FLAT:         ]
//
// Catch (int): bind e, end_catch, merge to return.
// CIR-FLAT:       ^[[CATCH_INT]](%{{.*}}: !cir.eh_token):
// CIR-FLAT:         %{{.*}}, %[[EXN_PTR:.*]] = cir.begin_catch %{{.*}} : !cir.eh_token -> (!cir.catch_token, !cir.ptr<!s32i>)
// CIR-FLAT:         cir.load{{.*}} %[[EXN_PTR]] : !cir.ptr<!s32i>, !s32i
// CIR-FLAT:         cir.store{{.*}} %{{.*}}, %[[E]] : !s32i, !cir.ptr<!s32i>
// CIR-FLAT:         cir.end_catch %{{.*}} : !cir.catch_token
// CIR-FLAT:         cir.br ^{{.*}}
//
// Catch-all handler.
// CIR-FLAT:       ^[[CATCH_ALL]](%[[CA_ET:.*]]: !cir.eh_token):
// CIR-FLAT:         %{{.*}}, %{{.*}} = cir.begin_catch %[[CA_ET]]
// CIR-FLAT:         cir.br ^{{.*}}
// CIR-FLAT:         cir.end_catch %{{.*}} : !cir.catch_token
// CIR-FLAT:         cir.br ^{{.*}}
//
// CIR-FLAT:         cir.return

// Lowering from CIR lists only the typed catch in each landingpad; the
// catch-all is reached from the shared dispatch block (typeid / icmp), not from
// a separate "catch ptr null" clause on those pads.

// LLVM-LABEL: define {{.*}} void @_Z40test_catch_all_and_specific_with_cleanupv()
// LLVM-SAME:    personality ptr @__gxx_personality_v0
// LLVM:         %[[S:.*]] = alloca %struct.S
// LLVM:         %[[E:.*]] = alloca i32
// LLVM:         invoke void @_ZN1SC1Ev({{.*}}%[[S]])
// LLVM:                 to label %[[AFTER_CTOR:.*]] unwind label %[[CTOR_LP:.*]]
// LLVM:       [[AFTER_CTOR]]:
// LLVM:         invoke void @_Z8mayThrowv()
// LLVM:                 to label %[[NORMAL:.*]] unwind label %[[INNER_LP:.*]]
// LLVM:       [[NORMAL]]:
// LLVM:         call void @_ZN1SD1Ev({{.*}}%[[S]])
// LLVM:       [[INNER_LP]]:
// LLVM:         landingpad { ptr, i32 }
// LLVM:                 catch ptr @_ZTIi
// LLVM:                 catch ptr null
// LLVM:         call void @_ZN1SD1Ev({{.*}}%[[S]])
// LLVM:       [[CTOR_LP]]:
// LLVM:         landingpad { ptr, i32 }
// LLVM:                 catch ptr @_ZTIi
// LLVM:                 catch ptr null
// LLVM:         call i32 @llvm.eh.typeid.for.p0(ptr @_ZTIi)
// LLVM:         icmp eq i32 {{.*}}, {{.*}}
// LLVM:       {{.*}}:
// LLVM:         call ptr @__cxa_begin_catch
// LLVM:         load i32, ptr {{.*}}
// LLVM:         store i32 {{.*}}, ptr %[[E]]
// LLVM:         call void @__cxa_end_catch()
// LLVM:       {{.*}}:
// LLVM:         call ptr @__cxa_begin_catch
// LLVM:         call void @__cxa_end_catch()
// LLVM:         ret void

// OGCG-LABEL: define {{.*}} void @_Z40test_catch_all_and_specific_with_cleanupv()
// OGCG-SAME:    personality ptr @__gxx_personality_v0
// OGCG:         %[[S:.*]] = alloca %struct.S
// OGCG:         invoke void @_ZN1SC1Ev({{.*}}%[[S]])
// OGCG:                 to label %[[AFTER_CTOR:.*]] unwind label %[[CTOR_LP:.*]]
// OGCG:       [[AFTER_CTOR]]:
// OGCG:         invoke void @_Z8mayThrowv()
// OGCG:                 to label %[[NORMAL:.*]] unwind label %[[INNER_LP:.*]]
// OGCG:       [[NORMAL]]:
// OGCG:         call void @_ZN1SD1Ev({{.*}}%[[S]])
// OGCG:       [[CTOR_LP]]:
// OGCG:         landingpad { ptr, i32 }
// OGCG:                 catch ptr @_ZTIi
// OGCG:                 catch ptr null
// OGCG:       [[INNER_LP]]:
// OGCG:         landingpad { ptr, i32 }
// OGCG:                 catch ptr @_ZTIi
// OGCG:                 catch ptr null
// OGCG:         call void @_ZN1SD1Ev({{.*}}%[[S]])
// OGCG:         call ptr @__cxa_begin_catch
// OGCG:         load i32, ptr {{.*}}
// OGCG:         store i32 {{.*}}, ptr %e
// OGCG:         call void @__cxa_end_catch()
// OGCG:         br label %try.cont
// OGCG:       try.cont:
// OGCG:         ret void
// OGCG:       catch:
// OGCG:         load ptr, ptr {{.*}}
// OGCG:         call ptr @__cxa_begin_catch
// OGCG:         call void @__cxa_end_catch()
// OGCG:         br label %try.cont
