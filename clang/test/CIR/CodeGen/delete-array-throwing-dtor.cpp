// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fcxx-exceptions -fexceptions -fclangir -emit-cir -mmlir -mlir-print-ir-after=cir-cxxabi-lowering -mmlir -mlir-print-ir-before=cir-cxxabi-lowering -mmlir -mlir-print-ir-after=cir-cxxabi-lowering %s -o %t.cir 2> %t-cxxabi.cir
// RUN: FileCheck --input-file=%t-cxxabi.cir --check-prefix=CIR-BEFORE-CXXABI %s
// RUN: FileCheck --input-file=%t-cxxabi.cir --check-prefix=CIR-AFTER-CXXABI %s
// RUN: FileCheck --input-file=%t.cir --check-prefix=CIR %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fcxx-exceptions -fexceptions -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll --check-prefix=LLVM %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fcxx-exceptions -fexceptions -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll --check-prefix=OGCG %s

struct ThrowingDtor {
  ~ThrowingDtor() noexcept(false);
  int x;
};

void test_delete_array_throwing_dtor(ThrowingDtor *ptr) {
  delete[] ptr;
}

// CIR-BEFORE-CXXABI: IR Dump Before CXXABILowering (cir-cxxabi-lowering)

// CIR-BEFORE-CXXABI: cir.func {{.*}} @_Z31test_delete_array_throwing_dtorP12ThrowingDtor
// CIR-BEFORE-CXXABI:   %[[PTR:.*]] = cir.load
// CIR-BEFORE-CXXABI:   %[[NULL:.*]] = cir.const #cir.ptr<null>
// CIR-BEFORE-CXXABI:   %[[NOT_NULL:.*]] = cir.cmp ne %[[PTR]], %[[NULL]]
// CIR-BEFORE-CXXABI:   cir.if %[[NOT_NULL]] {
// CIR-BEFORE-CXXABI:     cir.delete_array %[[PTR]] : !cir.ptr<!rec_ThrowingDtor> dtor_may_throw {delete_fn = @_ZdaPvm, delete_params = #cir.usual_delete_params<size = true>, element_dtor = @_ZN12ThrowingDtorD1Ev}
// CIR-BEFORE-CXXABI:   }

// CIR-AFTER-CXXABI: IR Dump After CXXABILowering (cir-cxxabi-lowering)

// CIR-AFTER-CXXABI: cir.func {{.*}} @_Z31test_delete_array_throwing_dtorP12ThrowingDtor
// CIR-AFTER-CXXABI:   %[[PTR:.*]] = cir.load
// CIR-AFTER-CXXABI:   %[[NULL:.*]] = cir.const #cir.ptr<null>
// CIR-AFTER-CXXABI:   %[[NOT_NULL:.*]] = cir.cmp ne %[[PTR]], %[[NULL]]
// CIR-AFTER-CXXABI:   cir.if %[[NOT_NULL]] {
// CIR-AFTER-CXXABI:     cir.cleanup.scope {
// CIR-AFTER-CXXABI:       cir.array.dtor %{{.*}}, %{{.*}} : !cir.ptr<!rec_ThrowingDtor>, !u64i dtor_may_throw {
// CIR-AFTER-CXXABI:         cir.call @_ZN12ThrowingDtorD1Ev({{.*}})
// CIR-AFTER-CXXABI-NOT:     nothrow
// CIR-AFTER-CXXABI:       }
// CIR-AFTER-CXXABI:     } cleanup all {
// CIR-AFTER-CXXABI:       cir.call @_ZdaPvm({{.*}}) nothrow
// CIR-AFTER-CXXABI:     }
// CIR-AFTER-CXXABI:   }

// CIR: cir.func {{.*}} @_Z31test_delete_array_throwing_dtorP12ThrowingDtor
// CIR:   %[[PTR:.*]] = cir.load
// CIR:   cir.if
//
// CIR:     %[[BYTE_PTR:.*]] = cir.cast bitcast %[[PTR]] : !cir.ptr<!rec_ThrowingDtor> -> !cir.ptr<!u8i>
// CIR:     %[[NEG_COOKIE:.*]] = cir.const #cir.int<-8> : !s64i
// CIR:     %[[ALLOC_BYTE_PTR:.*]] = cir.ptr_stride %[[BYTE_PTR]], %[[NEG_COOKIE]]
// CIR:     %[[VOID_PTR:.*]] = cir.cast bitcast %[[ALLOC_BYTE_PTR]] : !cir.ptr<!u8i> -> !cir.ptr<!void>
// CIR:     %[[COOKIE_PTR:.*]] = cir.cast bitcast %[[ALLOC_BYTE_PTR]] : !cir.ptr<!u8i> -> !cir.ptr<!u64i>
// CIR:     %[[NUM_ELEM:.*]] = cir.load{{.*}} %[[COOKIE_PTR]]
//
// CIR:     cir.cleanup.scope {
// CIR:       %[[END:.*]] = cir.ptr_stride %[[PTR]], %[[NUM_ELEM]]
// CIR:       %[[NOT_EMPTY:.*]] = cir.cmp ne %[[END]], %[[PTR]]
// CIR:       cir.if %[[NOT_EMPTY]] {
// CIR:         %[[ARR_IDX:.*]] = cir.alloca {{.*}} ["__array_idx"]
// CIR:         cir.store %[[END]], %[[ARR_IDX]]
//
// CIR:         cir.cleanup.scope {
// CIR:           cir.do {
// CIR:             %[[CUR:.*]] = cir.load %[[ARR_IDX]]
// CIR:             %[[STRIDE_M1:.*]] = cir.const #cir.int<-1> : !s64i
// CIR:             %[[PREV:.*]] = cir.ptr_stride %[[CUR]], %[[STRIDE_M1]]
// CIR:             cir.store %[[PREV]], %[[ARR_IDX]]
// CIR:             cir.call @_ZN12ThrowingDtorD1Ev(%[[PREV]])
// CIR-NOT:           nothrow
// CIR:             cir.yield
// CIR:           } while {
// CIR:             %[[CUR2:.*]] = cir.load %[[ARR_IDX]]
// CIR:             %[[CMP:.*]] = cir.cmp ne %[[CUR2]], %[[PTR]]
// CIR:             cir.condition(%[[CMP]])
// CIR:           }
// CIR:           cir.yield
// CIR:         } cleanup eh {
// CIR:           %[[CL_CUR:.*]] = cir.load %[[ARR_IDX]]
// CIR:           %[[CL_NEMPTY:.*]] = cir.cmp ne %[[CL_CUR]], %[[PTR]]
// CIR:           cir.if %[[CL_NEMPTY]] {
// CIR:             cir.do {
// CIR:               %[[CL_E:.*]] = cir.load %[[ARR_IDX]]
// CIR:               %[[CL_M1:.*]] = cir.const #cir.int<-1> : !s64i
// CIR:               %[[CL_PREV:.*]] = cir.ptr_stride %[[CL_E]], %[[CL_M1]]
// CIR:               cir.store %[[CL_PREV]], %[[ARR_IDX]]
// CIR:               cir.call @_ZN12ThrowingDtorD1Ev(%[[CL_PREV]])
// CIR-NOT:             nothrow
// CIR:             } while {
// CIR:             }
// CIR:           }
// CIR:           cir.yield
// CIR:         }
// CIR:       }
// CIR:       cir.yield
// CIR:     } cleanup all {
// CIR:       %[[ELEM_SIZE:.*]] = cir.const #cir.int<4> : !u64i
// CIR:       %[[ARRAY_SIZE:.*]] = cir.mul %[[ELEM_SIZE]], %[[NUM_ELEM]]
// CIR:       %[[COOKIE_SIZE:.*]] = cir.const #cir.int<8> : !u64i
// CIR:       %[[TOTAL_SIZE:.*]] = cir.add %[[ARRAY_SIZE]], %[[COOKIE_SIZE]]
// CIR:       cir.call @_ZdaPvm(%[[VOID_PTR]], %[[TOTAL_SIZE]]) nothrow
// CIR:       cir.yield
// CIR:     }
// CIR:   }

// LLVM: define {{.*}} void @_Z31test_delete_array_throwing_dtorP12ThrowingDtor(ptr {{.*}})
//
// `__array_idx` alloca and the function-entry null check.
// LLVM:   %[[ARR_IDX:[0-9]+]] = alloca ptr
// LLVM:   %[[PTR:.*]] = load ptr
// LLVM:   %[[NOT_NULL:.*]] = icmp ne ptr %[[PTR]], null
// LLVM:   br i1 %[[NOT_NULL]], label %[[NOTNULL:[^,]+]], label %{{.*}}
//
// Cookie read and is-empty check.
// LLVM: [[NOTNULL]]:
// LLVM:   %[[ALLOC_PTR:.*]] = getelementptr i8, ptr %[[PTR]], i64 -8
// LLVM:   %[[NUM_ELEM:.*]] = load i64, ptr %[[ALLOC_PTR]]
// LLVM:   %[[ARR_END:.*]] = getelementptr %struct.ThrowingDtor, ptr %[[PTR]], i64 %[[NUM_ELEM]]
// LLVM:   %[[NOT_EMPTY:.*]] = icmp ne ptr %[[ARR_END]], %[[PTR]]
// LLVM:   br i1 %[[NOT_EMPTY]], label %[[DESTROY:[^,]+]], label %[[CALL_DELETE_NORMAL:[^ ]+]]
//
// Body loop entry: seed __array_idx with arrayEnd and fall through into the
// body. (FlattenCFG emits an empty trampoline block between the entry
// store and the body, which we skip with `{{.*}}`.)
// LLVM: [[DESTROY]]:
// LLVM:   store ptr %[[ARR_END]], ptr %[[ARR_IDX]]
// LLVM:   br label %{{.*}}
//
// Body do-while condition block: load __array_idx, compare to begin, branch
// back to the body or out to the body-loop exit.
// LLVM:   %[[BODY_CUR:.*]] = load ptr, ptr %[[ARR_IDX]]
// LLVM:   %[[BODY_CMP:.*]] = icmp ne ptr %[[BODY_CUR]], %[[PTR]]
// LLVM:   br i1 %[[BODY_CMP]], label %[[BODY:[^,]+]], label %[[BODY_EXIT:[^ ]+]]
//
// Body do-while body: load current, compute prev = current - 1, store prev
// back to __array_idx, invoke dtor(prev). On unwind, go to LPAD.
// LLVM: [[BODY]]:
// LLVM:   %[[BODY_LOAD:.*]] = load ptr, ptr %[[ARR_IDX]]
// LLVM:   %[[BODY_PREV:.*]] = getelementptr %struct.ThrowingDtor, ptr %[[BODY_LOAD]], i64 -1
// LLVM:   store ptr %[[BODY_PREV]], ptr %[[ARR_IDX]]
// LLVM:   invoke void @_ZN12ThrowingDtorD1Ev(ptr %[[BODY_PREV]])
// LLVM:           to label %{{.*}} unwind label %[[LPAD:[^ ]+]]
//
// Cleanup landing pad: cleanup landingpad, save exn/sel, then check whether
// any elements remain to destroy. (FlattenCFG emits a trampoline block
// from the landingpad to the phi-of-exn/sel and the empty-check branch.)
// LLVM: [[LPAD]]:
// LLVM:   %[[LPAD_VAL:.*]] = landingpad { ptr, i32 }
// LLVM:           cleanup
// LLVM:   %[[LPAD_EXN:.*]] = extractvalue { ptr, i32 } %[[LPAD_VAL]], 0
// LLVM:   %[[LPAD_SEL:.*]] = extractvalue { ptr, i32 } %[[LPAD_VAL]], 1
// LLVM:   br label %{{.*}}
//
// LPAD continuation: phi the exn/sel forward, load __array_idx, check for
// empty, and branch into the cleanup loop or to the EH-side delete.
// LLVM:   phi ptr [ %[[LPAD_EXN]], %{{.*}} ]
// LLVM:   phi i32 [ %[[LPAD_SEL]], %{{.*}} ]
// LLVM:   %[[CL_INIT_CUR:.*]] = load ptr, ptr %[[ARR_IDX]]
// LLVM:   %[[CL_NEMPTY:.*]] = icmp ne ptr %[[CL_INIT_CUR]], %[[PTR]]
// LLVM:   br i1 %[[CL_NEMPTY]], label %{{.*}}, label %{{.*}}
//
// Cleanup do-while condition block: load __array_idx (which the body
// already pointed at the element that threw, so the cleanup picks up at
// prev = element-that-threw - 1), compare to begin, branch into the
// cleanup body or out to the EH-side delete.
// LLVM:   %[[CL_CUR:.*]] = load ptr, ptr %[[ARR_IDX]]
// LLVM:   %[[CL_CMP:.*]] = icmp ne ptr %[[CL_CUR]], %[[PTR]]
// LLVM:   br i1 %[[CL_CMP]], label %[[CL_BODY:[^,]+]], label %{{.*}}
//
// Cleanup do-while body: load current, decrement, store back, invoke
// dtor(prev). On a *second* throw, unwind to terminate.lpad.
// LLVM: [[CL_BODY]]:
// LLVM:   %[[CL_LOAD:.*]] = load ptr, ptr %[[ARR_IDX]]
// LLVM:   %[[CL_PREV:.*]] = getelementptr %struct.ThrowingDtor, ptr %[[CL_LOAD]], i64 -1
// LLVM:   store ptr %[[CL_PREV]], ptr %[[ARR_IDX]]
// LLVM:   invoke void @_ZN12ThrowingDtorD1Ev(ptr %[[CL_PREV]])
// LLVM:           to label %{{.*}} unwind label %[[TERMINATE_LPAD:[^ ]+]]
//
// Terminate landing pad: catch-all + `__clang_call_terminate`.
// LLVM: [[TERMINATE_LPAD]]:
// LLVM:   landingpad { ptr, i32 }
// LLVM:           catch ptr null
// LLVM:   call void @__clang_call_terminate(ptr %{{.*}})
// LLVM:   unreachable
//
// Normal path: compute total size, call `_ZdaPvm`. Mirrors OGCG's
// `arraydestroy.done8` block.
// LLVM: [[CALL_DELETE_NORMAL]]:
// LLVM:   %[[NORMAL_ARRAY_SIZE:.*]] = mul i64 4, %[[NUM_ELEM]]
// LLVM:   %[[NORMAL_TOTAL_SIZE:.*]] = add i64 %[[NORMAL_ARRAY_SIZE]], 8
// LLVM:   call void @_ZdaPvm(ptr %[[ALLOC_PTR]], i64 %[[NORMAL_TOTAL_SIZE]])
//
// EH-side delete + resume: the exn/sel are PHI'd one more time as
// FlattenCFG joins the cleanup-loop exit blocks; then total size is
// computed, `_ZdaPvm` is called, and the original exception is resumed.
// Mirrors OGCG's `arraydestroy.done6` -> `eh.resume` chain.
// LLVM:   %[[RESUME_EXN:.*]] = phi ptr [ %{{.*}}, %{{.*}} ]
// LLVM:   %[[RESUME_SEL:.*]] = phi i32 [ %{{.*}}, %{{.*}} ]
// LLVM:   %[[EH_ARRAY_SIZE:.*]] = mul i64 4, %[[NUM_ELEM]]
// LLVM:   %[[EH_TOTAL_SIZE:.*]] = add i64 %[[EH_ARRAY_SIZE]], 8
// LLVM:   call void @_ZdaPvm(ptr %[[ALLOC_PTR]], i64 %[[EH_TOTAL_SIZE]])
// LLVM:   %[[RESUME_VAL:.*]] = insertvalue { ptr, i32 } poison, ptr %[[RESUME_EXN]], 0
// LLVM:   %[[RESUME_VAL2:.*]] = insertvalue { ptr, i32 } %[[RESUME_VAL]], i32 %[[RESUME_SEL]], 1
// LLVM:   resume { ptr, i32 } %[[RESUME_VAL2]]

// OGCG: define {{.*}} void @_Z31test_delete_array_throwing_dtorP12ThrowingDtor(ptr {{.*}})
//
// Function entry and null check.
// OGCG:   %[[PTR:.*]] = load ptr, ptr %{{.*}}
// OGCG:   %[[ISNULL:.*]] = icmp eq ptr %[[PTR]], null
// OGCG:   br i1 %[[ISNULL]], label %[[DELETE_END:[^,]+]], label %[[DELETE_NOTNULL:[^ ]+]]
//
// Cookie read + is-empty check.
// OGCG: [[DELETE_NOTNULL]]:
// OGCG:   %[[ALLOC_PTR:.*]] = getelementptr inbounds i8, ptr %[[PTR]], i64 -8
// OGCG:   %[[NUM_ELEM:.*]] = load i64, ptr %[[ALLOC_PTR]]
// OGCG:   %[[ARR_END:.*]] = getelementptr inbounds %struct.ThrowingDtor, ptr %[[PTR]], i64 %[[NUM_ELEM]]
// OGCG:   %[[ISEMPTY:.*]] = icmp eq ptr %[[PTR]], %[[ARR_END]]
// OGCG:   br i1 %[[ISEMPTY]], label %[[DONE8:[^,]+]], label %[[BODY:[^ ]+]]
//
// Body loop: phi-based reverse iteration.
// OGCG: [[BODY]]:
// OGCG:   %[[ELT_PAST:.*]] = phi ptr [ %[[ARR_END]], %[[DELETE_NOTNULL]] ], [ %[[ELT:.*]], %[[INV_CONT:[^ ]+]] ]
// OGCG:   %[[ELT]] = getelementptr inbounds %struct.ThrowingDtor, ptr %[[ELT_PAST]], i64 -1
// OGCG:   invoke void @_ZN12ThrowingDtorD1Ev(ptr {{.*}}%[[ELT]])
// OGCG:           to label %[[INV_CONT]] unwind label %[[LPAD:[^ ]+]]
//
// OGCG: [[INV_CONT]]:
// OGCG:   %[[BODY_DONE:.*]] = icmp eq ptr %[[ELT]], %[[PTR]]
// OGCG:   br i1 %[[BODY_DONE]], label %[[DONE8]], label %[[BODY]]
//
// Normal path: compute size, call `_ZdaPvm`, fall through.
// OGCG: [[DONE8]]:
// OGCG:   %[[NORMAL_ARRAY_SIZE:.*]] = mul i64 4, %[[NUM_ELEM]]
// OGCG:   %[[NORMAL_TOTAL_SIZE:.*]] = add i64 %[[NORMAL_ARRAY_SIZE]], 8
// OGCG:   call void @_ZdaPvm(ptr {{.*}}%[[ALLOC_PTR]], i64 {{.*}}%[[NORMAL_TOTAL_SIZE]])
// OGCG:   br label %[[DELETE_END]]
//
// OGCG: [[DELETE_END]]:
// OGCG:   ret void
//
// Cleanup landing pad: cleanup landingpad, save exn/sel, then check whether
// any elements remain to destroy.
// OGCG: [[LPAD]]:
// OGCG:   %[[LPAD_VAL:.*]] = landingpad { ptr, i32 }
// OGCG:           cleanup
// OGCG:   %[[CL_ISEMPTY:.*]] = icmp eq ptr %[[PTR]], %[[ELT]]
// OGCG:   br i1 %[[CL_ISEMPTY]], label %[[DONE6:[^,]+]], label %[[BODY2:[^ ]+]]
//
// Cleanup loop: phi starts at the element that threw, decrements to the
// previous element, invokes dtor unwinding to terminate.lpad on a second
// throw.
// OGCG: [[BODY2]]:
// OGCG:   %[[ELT_PAST3:.*]] = phi ptr [ %[[ELT]], %[[LPAD]] ], [ %[[ELT4:.*]], %[[INV_CONT5:[^ ]+]] ]
// OGCG:   %[[ELT4]] = getelementptr inbounds %struct.ThrowingDtor, ptr %[[ELT_PAST3]], i64 -1
// OGCG:   invoke void @_ZN12ThrowingDtorD1Ev(ptr {{.*}}%[[ELT4]])
// OGCG:           to label %[[INV_CONT5]] unwind label %[[TERMINATE_LPAD:[^ ]+]]
//
// OGCG: [[INV_CONT5]]:
// OGCG:   %[[CL_DONE:.*]] = icmp eq ptr %[[ELT4]], %[[PTR]]
// OGCG:   br i1 %[[CL_DONE]], label %[[DONE6]], label %[[BODY2]]
//
// EH path: compute size, call `_ZdaPvm`, resume.
// OGCG: [[DONE6]]:
// OGCG:   %[[EH_ARRAY_SIZE:.*]] = mul i64 4, %[[NUM_ELEM]]
// OGCG:   %[[EH_TOTAL_SIZE:.*]] = add i64 %[[EH_ARRAY_SIZE]], 8
// OGCG:   call void @_ZdaPvm(ptr {{.*}}%[[ALLOC_PTR]], i64 {{.*}}%[[EH_TOTAL_SIZE]])
// OGCG:   br label %[[EH_RESUME:[^ ]+]]
//
// OGCG: [[EH_RESUME]]:
// OGCG:   resume { ptr, i32 }
//
// Terminate landing pad.
// OGCG: [[TERMINATE_LPAD]]:
// OGCG:   landingpad { ptr, i32 }
// OGCG:           catch ptr null
// OGCG:   call void @__clang_call_terminate(ptr {{.*}})
// OGCG:   unreachable
