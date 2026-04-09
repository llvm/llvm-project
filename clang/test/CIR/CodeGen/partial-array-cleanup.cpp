// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -fexceptions -fcxx-exceptions -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir  2> %t-before-lp.cir
// RUN: FileCheck --input-file=%t-before-lp.cir %s -check-prefix=CIR-BEFORE-LPP
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fexceptions -fcxx-exceptions -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fexceptions -fcxx-exceptions -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

struct S {
    S();
    ~S();
};

void test_partial_array_cleanup() {
    S s[4];
}

// CIR-BEFORE-LPP:     cir.func {{.*}} @_Z26test_partial_array_cleanupv()
// CIR-BEFORE-LPP:       %[[ARRAY:.*]] = cir.alloca !cir.array<!rec_S x 4>, !cir.ptr<!cir.array<!rec_S x 4>>, ["s", init]
// CIR-BEFORE-LPP:       cir.array.ctor %[[ARRAY]] : !cir.ptr<!cir.array<!rec_S x 4>> {
// CIR-BEFORE-LPP:       ^bb0(%[[CTOR_ARG:.*]]: !cir.ptr<!rec_S>):
// CIR-BEFORE-LPP:         cir.call @_ZN1SC1Ev(%[[CTOR_ARG]]) : (!cir.ptr<!rec_S>{{.*}}) -> ()
// CIR-BEFORE-LPP:       } partial_dtor {
// CIR-BEFORE-LPP:       ^bb0(%[[DTOR_ARG:.*]]: !cir.ptr<!rec_S>):
// CIR-BEFORE-LPP:         cir.call @_ZN1SD1Ev(%[[DTOR_ARG]]){{.*}} : (!cir.ptr<!rec_S>{{.*}}) -> ()
// CIR-BEFORE-LPP:       }

// CIR:     cir.func {{.*}} @_Z26test_partial_array_cleanupv()
// CIR:       %[[ARRAY:.*]] = cir.alloca !cir.array<!rec_S x 4>, !cir.ptr<!cir.array<!rec_S x 4>>, ["s", init]
// CIR:       %[[CONST4:.*]] = cir.const #cir.int<4> : !u64i
// CIR:       %[[BEGIN:.*]] = cir.cast array_to_ptrdecay %[[ARRAY]] : !cir.ptr<!cir.array<!rec_S x 4>> -> !cir.ptr<!rec_S>
// CIR:       %[[END:.*]] = cir.ptr_stride %[[BEGIN]], %[[CONST4]] : (!cir.ptr<!rec_S>, !u64i) -> !cir.ptr<!rec_S>
// CIR:       %[[ITER:.*]] = cir.alloca !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>, ["__array_idx"]
// CIR:       cir.store %[[BEGIN]], %[[ITER]] : !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>
// CIR:       cir.cleanup.scope {
// CIR:         cir.do {
// CIR:           %[[CUR:.*]] = cir.load %[[ITER]] : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
// CIR:           cir.call @_ZN1SC1Ev(%[[CUR]]) : (!cir.ptr<!rec_S>{{.*}}) -> ()
// CIR:           %[[CONST1:.*]] = cir.const #cir.int<1> : !u64i
// CIR:           %[[NEXT:.*]] = cir.ptr_stride %[[CUR]], %[[CONST1]] : (!cir.ptr<!rec_S>, !u64i) -> !cir.ptr<!rec_S>
// CIR:           cir.store %[[NEXT]], %[[ITER]] : !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>
// CIR:           cir.yield
// CIR:         } while {
// CIR:           %[[CUR2:.*]] = cir.load %[[ITER]] : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
// CIR:           %[[CMP:.*]] = cir.cmp ne %[[CUR2]], %[[END]] : !cir.ptr<!rec_S>
// CIR:           cir.condition(%[[CMP]])
// CIR:         }
// CIR:         cir.yield
// CIR:       } cleanup eh {
// CIR:         %[[CUR3:.*]] = cir.load %[[ITER]] : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
// CIR:         %[[NE:.*]] = cir.cmp ne %[[CUR3]], %[[BEGIN]] : !cir.ptr<!rec_S>
// CIR:         cir.if %[[NE]] {
// CIR:           cir.do {
// CIR:             %[[EL:.*]] = cir.load %[[ITER]] : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
// CIR:             %[[NEG1:.*]] = cir.const #cir.int<-1> : !s64i
// CIR:             %[[PREV:.*]] = cir.ptr_stride %[[EL]], %[[NEG1]] : (!cir.ptr<!rec_S>, !s64i) -> !cir.ptr<!rec_S>
// CIR:             cir.store %[[PREV]], %[[ITER]] : !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>
// CIR:             cir.call @_ZN1SD1Ev(%[[PREV]]){{.*}} : (!cir.ptr<!rec_S>{{.*}}) -> ()
// CIR:             cir.yield
// CIR:           } while {
// CIR:             %[[EL2:.*]] = cir.load %[[ITER]] : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
// CIR:             %[[NE2:.*]] = cir.cmp ne %[[EL2]], %[[BEGIN]] : !cir.ptr<!rec_S>
// CIR:             cir.condition(%[[NE2]])
// CIR:           }
// CIR:         }
// CIR:         cir.yield
// CIR:       }

// LLVM:     define dso_local void @_Z26test_partial_array_cleanupv()
// LLVM:       %[[ARRAY:.*]] = alloca [4 x %struct.S]
// LLVM:       %[[BEGIN:.*]] = getelementptr %struct.S, ptr %[[ARRAY]], i32 0
// LLVM:       %[[END:.*]] = getelementptr %struct.S, ptr %[[BEGIN]], i64 4
// LLVM:       %[[ITER:.*]] = alloca ptr
// LLVM:       store ptr %[[BEGIN]], ptr %[[ITER]]
//
//            --- ctor loop condition ---
// LLVM:     [[CLEANUP_SCOPE:.*]]:
// LLVM:       br label %[[CTOR_LOOP_COND:.*]]
//
// LLVM:     [[CTOR_LOOP_COND:]]:
// LLVM:       %[[COND_CUR:.*]] = load ptr, ptr %[[ITER]]
// LLVM:       %[[CTOR_DONE:.*]] = icmp ne ptr %[[COND_CUR]], %[[END]]
// LLVM:       br i1 %[[CTOR_DONE]], label %[[CTOR_BODY:.*]], label %[[CTOR_EXIT:.*]]
//
//            --- ctor loop body ---
// LLVM:     [[CTOR_BODY]]:
// LLVM:       %[[CUR:.*]] = load ptr, ptr %[[ITER]]
// LLVM:       invoke void @_ZN1SC1Ev(ptr{{.*}} %[[CUR]])
// LLVM:         to label %[[CTOR_CONT:.*]] unwind label %[[LPAD:.*]]
//
// LLVM:     [[CTOR_CONT]]:
// LLVM:       %[[NEXT:.*]] = getelementptr %struct.S, ptr %[[CUR]], i64 1
// LLVM:       store ptr %[[NEXT]], ptr %[[ITER]]
// LLVM:       br label %[[CTOR_LOOP_COND]]
//
//            --- landing pad + cleanup guard ---
// LLVM:     [[LPAD]]:
// LLVM:       landingpad { ptr, i32 }
// LLVM:         cleanup
// LLVM:       %[[PAD_CUR:.*]] = load ptr, ptr %[[ITER]]
// LLVM:       %[[GUARD:.*]] = icmp ne ptr %[[PAD_CUR]], %[[BEGIN]]
// LLVM:       br i1 %[[GUARD]], label %[[DTOR_ENTRY:.*]], label %[[EH_RESUME:.*]]
//
//            --- partial dtor do-while entry ---
// LLVM:     [[DTOR_ENTRY]]:
// LLVM:       br label %[[DTOR_BODY:.*]]
//
//            --- partial dtor loop condition (back-edge) ---
// LLVM:     [[DTOR_LOOP_COND:.*]]:
// LLVM:       %[[DTOR_CUR:.*]] = load ptr, ptr %[[ITER]]
// LLVM:       %[[DTOR_CONT:.*]] = icmp ne ptr %[[DTOR_CUR]], %[[BEGIN]]
// LLVM:       br i1 %[[DTOR_CONT]], label %[[DTOR_BODY]], label %[[DTOR_DONE:.*]]
//
//            --- partial dtor loop body ---
// LLVM:     [[DTOR_BODY]]:
// LLVM:       %[[DCUR:.*]] = load ptr, ptr %[[ITER]]
// LLVM:       %[[PREV:.*]] = getelementptr %struct.S, ptr %[[DCUR]], i64 -1
// LLVM:       store ptr %[[PREV]], ptr %[[ITER]]
// LLVM:       call void @_ZN1SD1Ev(ptr{{.*}} %[[PREV]])
// LLVM:       br label %[[DTOR_LOOP_COND]]
//
// LLVM:     [[DTOR_DONE]]:
// LLVM:       br label %[[EH_RESUME]]
//
// LLVM:     [[EH_RESUME]]:
// LLVM:       resume { ptr, i32 }

// OGCG:     define dso_local void @_Z26test_partial_array_cleanupv()
// OGCG:     [[ENTRY:.*]]:
// OGCG:       %[[ARRAY:.*]] = alloca [4 x %struct.S]
// OGCG:       %[[BEGIN:.*]] = getelementptr inbounds [4 x %struct.S], ptr %[[ARRAY]], i32 0, i32 0
// OGCG:       %[[END:.*]] = getelementptr inbounds %struct.S, ptr %[[BEGIN]], i64 4
//
//            --- ctor loop ---
// OGCG:     [[CTOR_BODY:.*]]:
// OGCG:       %[[CUR:.*]] = phi ptr [ %[[BEGIN]], %[[ENTRY]] ], [ %[[NEXT:.*]], %[[CONT:.*]] ]
// OGCG:       invoke void @_ZN1SC1Ev(ptr{{.*}})
// OGCG:         to label %[[CONT]] unwind label %[[LPAD:.*]]
//
// OGCG:     [[CONT]]:
// OGCG:       %[[NEXT]] = getelementptr inbounds %struct.S, ptr %[[CUR]], i64 1
// OGCG:       %[[DONE:.*]] = icmp eq ptr %[[NEXT]], %[[END]]
// OGCG:       br i1 %[[DONE]], label %[[CTOR_EXIT:.*]], label %[[CTOR_BODY:.*]]
//
//            --- landing pad + cleanup guard ---
// OGCG:     [[LPAD]]:
// OGCG:       landingpad { ptr, i32 }
// OGCG:         cleanup
// OGCG:       %[[ISEMPTY:.*]] = icmp eq ptr %[[BEGIN]], %[[CUR]]
// OGCG:       br i1 %[[ISEMPTY]], label %[[EH_RESUME:.*]], label %[[DTOR_LOOP:.*]]
//
//            --- partial dtor loop ---
// OGCG:     [[DTOR_LOOP]]:
// OGCG:       %[[PAST:.*]] = phi ptr [ %[[CUR]], %[[LPAD]] ], [ %[[PREV:.*]], %[[DTOR_LOOP]] ]
// OGCG:       %[[PREV]] = getelementptr inbounds %struct.S, ptr %[[PAST]], i64 -1
// OGCG:       call void @_ZN1SD1Ev(ptr{{.*}} %[[PREV]])
// OGCG:       %[[DDONE:.*]] = icmp eq ptr %[[PREV]], %[[BEGIN]]
// OGCG:       br i1 %[[DDONE]], label %[[EH_RESUME]], label %[[DTOR_LOOP]]
//
// OGCG:     [[EH_RESUME]]:
// OGCG:       resume { ptr, i32 }
