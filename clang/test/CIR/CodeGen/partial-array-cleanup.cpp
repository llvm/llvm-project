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

void test_variable_size_partial_array_cleanup(int n) {
    S s[n];
}

// CIR-BEFORE-LPP:     cir.func {{.*}} @_Z40test_variable_size_partial_array_cleanupi
// CIR-BEFORE-LPP:       %[[N_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["n", init]
// CIR-BEFORE-LPP:       %[[N_VAL:.*]] = cir.load {{.*}} %[[N_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR-BEFORE-LPP:       %[[N:.*]] = cir.cast integral %[[N_VAL]] : !s32i -> !u64i
// CIR-BEFORE-LPP:       cir.stacksave
// CIR-BEFORE-LPP:       cir.cleanup.scope {
// CIR-BEFORE-LPP:         %[[VLA:.*]] = cir.alloca !rec_S, !cir.ptr<!rec_S>, %[[N]] : !u64i, ["s", init]
// CIR-BEFORE-LPP:         cir.array.ctor %[[VLA]], %[[N]] : !cir.ptr<!rec_S>, !u64i {
// CIR-BEFORE-LPP:         ^bb0(%[[CTOR_ARG:.*]]: !cir.ptr<!rec_S>):
// CIR-BEFORE-LPP:           cir.call @_ZN1SC1Ev(%[[CTOR_ARG]]) : (!cir.ptr<!rec_S>{{.*}}) -> ()
// CIR-BEFORE-LPP:         } partial_dtor {
// CIR-BEFORE-LPP:         ^bb0(%[[DTOR_ARG:.*]]: !cir.ptr<!rec_S>):
// CIR-BEFORE-LPP:           cir.call @_ZN1SD1Ev(%[[DTOR_ARG]]){{.*}} : (!cir.ptr<!rec_S>{{.*}}) -> ()
// CIR-BEFORE-LPP:         }
// CIR-BEFORE-LPP:         cir.cleanup.scope {
// CIR-BEFORE-LPP:           cir.yield
// CIR-BEFORE-LPP:         } cleanup all {
// CIR-BEFORE-LPP:           cir.array.dtor %[[VLA]], %[[N]] : !cir.ptr<!rec_S>, !u64i {
// CIR-BEFORE-LPP:           ^bb0(%[[DTOR_ARG2:.*]]: !cir.ptr<!rec_S>):
// CIR-BEFORE-LPP:             cir.call @_ZN1SD1Ev(%[[DTOR_ARG2]]){{.*}} : (!cir.ptr<!rec_S>{{.*}}) -> ()
// CIR-BEFORE-LPP:           }
// CIR-BEFORE-LPP:       } cleanup normal {
// CIR-BEFORE-LPP:         cir.stackrestore
// CIR-BEFORE-LPP:       }

// CIR:     cir.func {{.*}} @_Z40test_variable_size_partial_array_cleanupi
// CIR:       %[[N_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["n", init]
// CIR:       %[[SAVED_STACK:.*]] = cir.alloca !cir.ptr<!u8i>, !cir.ptr<!cir.ptr<!u8i>>, ["saved_stack"]
// CIR:       %[[N_VAL:.*]] = cir.load {{.*}} %[[N_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:       %[[N:.*]] = cir.cast integral %[[N_VAL]] : !s32i -> !u64i
// CIR:       %[[STACK:.*]] = cir.stacksave : !cir.ptr<!u8i>
// CIR:       cir.store {{.*}} %[[STACK]], %[[SAVED_STACK]] : !cir.ptr<!u8i>, !cir.ptr<!cir.ptr<!u8i>>
// CIR:       cir.cleanup.scope {
// CIR:         %[[BEGIN:.*]] = cir.alloca !rec_S, !cir.ptr<!rec_S>, %[[N]] : !u64i, ["s", init]
// CIR:         %[[END:.*]] = cir.ptr_stride %[[BEGIN]], %[[N]] : (!cir.ptr<!rec_S>, !u64i) -> !cir.ptr<!rec_S>
// CIR:         %[[ZERO:.*]] = cir.const #cir.int<0> : !u64i
// CIR:         %[[IS_NONZERO:.*]] = cir.cmp ne %[[N]], %[[ZERO]] : !u64i
// CIR:         cir.if %[[IS_NONZERO]] {
// CIR:           %[[ITER:.*]] = cir.alloca !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>, ["__array_idx"]
// CIR:           cir.store %[[BEGIN]], %[[ITER]] : !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>
// CIR:           cir.cleanup.scope {
// CIR:             cir.do {
// CIR:               %[[CUR:.*]] = cir.load %[[ITER]] : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
// CIR:               cir.call @_ZN1SC1Ev(%[[CUR]]) : (!cir.ptr<!rec_S>{{.*}}) -> ()
// CIR:               %[[CONST1:.*]] = cir.const #cir.int<1> : !u64i
// CIR:               %[[NEXT:.*]] = cir.ptr_stride %[[CUR]], %[[CONST1]] : (!cir.ptr<!rec_S>, !u64i) -> !cir.ptr<!rec_S>
// CIR:               cir.store %[[NEXT]], %[[ITER]] : !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>
// CIR:               cir.yield
// CIR:             } while {
// CIR:               %[[CUR2:.*]] = cir.load %[[ITER]] : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
// CIR:               %[[CMP:.*]] = cir.cmp ne %[[CUR2]], %[[END]] : !cir.ptr<!rec_S>
// CIR:               cir.condition(%[[CMP]])
// CIR:             }
// CIR:             cir.yield
// CIR:           } cleanup eh {
// CIR:             %[[CUR3:.*]] = cir.load %[[ITER]] : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
// CIR:             %[[NE:.*]] = cir.cmp ne %[[CUR3]], %[[BEGIN]] : !cir.ptr<!rec_S>
// CIR:             cir.if %[[NE]] {
// CIR:               cir.do {
// CIR:                 %[[EL:.*]] = cir.load %[[ITER]] : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
// CIR:                 %[[NEG1:.*]] = cir.const #cir.int<-1> : !s64i
// CIR:                 %[[PREV:.*]] = cir.ptr_stride %[[EL]], %[[NEG1]] : (!cir.ptr<!rec_S>, !s64i) -> !cir.ptr<!rec_S>
// CIR:                 cir.store %[[PREV]], %[[ITER]] : !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>
// CIR:                 cir.call @_ZN1SD1Ev(%[[PREV]]){{.*}} : (!cir.ptr<!rec_S>{{.*}}) -> ()
// CIR:                 cir.yield
// CIR:               } while {
// CIR:                 %[[EL2:.*]] = cir.load %[[ITER]] : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
// CIR:                 %[[NE2:.*]] = cir.cmp ne %[[EL2]], %[[BEGIN]] : !cir.ptr<!rec_S>
// CIR:                 cir.condition(%[[NE2]])
// CIR:               }
// CIR:             }
// CIR:             cir.yield
// CIR:           }
// CIR:         }
//
//              --- normal dtor ---
// CIR:         } cleanup all {
// CIR:           %[[LAST:.*]] = cir.ptr_stride %[[BEGIN]], %[[N]] : (!cir.ptr<!rec_S>, !u64i) -> !cir.ptr<!rec_S>
// CIR:           %[[DTOR_NE:.*]] = cir.cmp ne %[[LAST]], %[[BEGIN]] : !cir.ptr<!rec_S>
// CIR:           cir.if %[[DTOR_NE]] {
// CIR:             %[[DTOR_ITER:.*]] = cir.alloca !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>, ["__array_idx"]
// CIR:             cir.store %[[LAST]], %[[DTOR_ITER]] : !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>
// CIR:             cir.do {
// CIR:               %[[DTOR_CUR:.*]] = cir.load %[[DTOR_ITER]] : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
// CIR:               %[[DTOR_NEG1:.*]] = cir.const #cir.int<-1> : !s64i
// CIR:               %[[DTOR_PREV:.*]] = cir.ptr_stride %[[DTOR_CUR]], %[[DTOR_NEG1]] : (!cir.ptr<!rec_S>, !s64i) -> !cir.ptr<!rec_S>
// CIR:               cir.store %[[DTOR_PREV]], %[[DTOR_ITER]] : !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>
// CIR:               cir.call @_ZN1SD1Ev(%[[DTOR_PREV]]){{.*}} : (!cir.ptr<!rec_S>{{.*}}) -> ()
// CIR:               cir.yield
// CIR:             } while {
// CIR:               %[[DTOR_CUR2:.*]] = cir.load %[[DTOR_ITER]] : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
// CIR:               %[[DTOR_NE2:.*]] = cir.cmp ne %[[DTOR_CUR2]], %[[BEGIN]] : !cir.ptr<!rec_S>
// CIR:               cir.condition(%[[DTOR_NE2]])
// CIR:             }
// CIR:           }
// CIR:           cir.yield
// CIR:         }
// CIR:         cir.yield
// CIR:       } cleanup normal {
// CIR:         cir.stackrestore {{.*}} : !cir.ptr<!u8i>
// CIR:         cir.yield
// CIR:       }

// LLVM:     define dso_local void @_Z40test_variable_size_partial_array_cleanupi
// LLVM:       %[[N_I32:.*]] = load i32, ptr %{{.*}}
// LLVM:       %[[N:.*]] = sext i32 %[[N_I32]] to i64
// LLVM:       call ptr @llvm.stacksave.p0()
//
//            --- VLA alloca + zero check ---
// LLVM:       %[[BEGIN:.*]] = alloca %struct.S, i64 %[[N]]
// LLVM:       %[[END:.*]] = getelementptr %struct.S, ptr %[[BEGIN]], i64 %[[N]]
// LLVM:       %[[IS_NONZERO:.*]] = icmp ne i64 %[[N]], 0
// LLVM:       br i1 %[[IS_NONZERO]], label %[[CTOR_SETUP:.*]], label %[[AFTER_CTOR:.*]]
//
//            --- ctor init ---
// LLVM:     [[CTOR_SETUP]]:
// LLVM:       store ptr %[[BEGIN]], ptr %[[CTOR_ITER:.*]]
// LLVM:       br label %[[CTOR_ENTRY:.*]]
//
// LLVM:     [[CTOR_ENTRY]]:
// LLVM:       br label %[[CTOR_BODY:.*]]
//
//            --- ctor loop condition (back-edge) ---
// LLVM:     [[CTOR_LOOP_COND:.*]]:
// LLVM:       %[[COND_CUR:.*]] = load ptr, ptr %[[CTOR_ITER]]
// LLVM:       %[[CTOR_DONE:.*]] = icmp ne ptr %[[COND_CUR]], %[[END]]
// LLVM:       br i1 %[[CTOR_DONE]], label %[[CTOR_BODY]], label %[[CTOR_EXIT:.*]]
//
//            --- ctor loop body ---
// LLVM:     [[CTOR_BODY]]:
// LLVM:       %[[CUR:.*]] = load ptr, ptr %[[CTOR_ITER]]
// LLVM:       invoke void @_ZN1SC1Ev(ptr{{.*}} %[[CUR]])
// LLVM:         to label %[[CTOR_CONT:.*]] unwind label %[[LPAD:.*]]
//
// LLVM:     [[CTOR_CONT]]:
// LLVM:       %[[NEXT:.*]] = getelementptr %struct.S, ptr %[[CUR]], i64 1
// LLVM:       store ptr %[[NEXT]], ptr %[[CTOR_ITER]]
// LLVM:       br label %[[CTOR_LOOP_COND]]
//
// LLVM:     [[CTOR_EXIT]]:
// LLVM:       br label %[[CTOR_JOIN:.*]]
//
//            --- landing pad + cleanup guard ---
// LLVM:     [[LPAD]]:
// LLVM:       landingpad { ptr, i32 }
// LLVM:         cleanup
// LLVM:       %[[PAD_CUR:.*]] = load ptr, ptr %[[CTOR_ITER]]
// LLVM:       %[[GUARD:.*]] = icmp ne ptr %[[PAD_CUR]], %[[BEGIN]]
// LLVM:       br i1 %[[GUARD]], label %[[PDTOR_ENTRY:.*]], label %[[EH_RESUME:.*]]
//
//            --- partial dtor do-while entry ---
// LLVM:     [[PDTOR_ENTRY]]:
// LLVM:       br label %[[PDTOR_BODY:.*]]
//
//            --- partial dtor loop condition (back-edge) ---
// LLVM:     [[PDTOR_LOOP_COND:.*]]:
// LLVM:       %[[PDTOR_CUR:.*]] = load ptr, ptr %[[CTOR_ITER]]
// LLVM:       %[[PDTOR_CONT:.*]] = icmp ne ptr %[[PDTOR_CUR]], %[[BEGIN]]
// LLVM:       br i1 %[[PDTOR_CONT]], label %[[PDTOR_BODY]], label %[[PDTOR_DONE:.*]]
//
//            --- partial dtor loop body ---
// LLVM:     [[PDTOR_BODY]]:
// LLVM:       %[[PDCUR:.*]] = load ptr, ptr %[[CTOR_ITER]]
// LLVM:       %[[PPREV:.*]] = getelementptr %struct.S, ptr %[[PDCUR]], i64 -1
// LLVM:       store ptr %[[PPREV]], ptr %[[CTOR_ITER]]
// LLVM:       call void @_ZN1SD1Ev(ptr{{.*}} %[[PPREV]])
// LLVM:       br label %[[PDTOR_LOOP_COND]]
//
// LLVM:     [[PDTOR_DONE]]:
// LLVM:       br label %[[EH_RESUME]]
//
// LLVM:     [[EH_RESUME]]:
// LLVM:       resume { ptr, i32 }
//
//            --- ctor done join ---
// LLVM:     [[CTOR_JOIN]]:
// LLVM:       br label %[[AFTER_CTOR]]
//
//            --- normal dtor setup ---
// LLVM:     [[AFTER_CTOR]]:
// LLVM:       %[[LAST:.*]] = getelementptr %struct.S, ptr %[[BEGIN]], i64 %[[N]]
// LLVM:       %[[DTOR_NE:.*]] = icmp ne ptr %[[LAST]], %[[BEGIN]]
// LLVM:       br i1 %[[DTOR_NE]], label %[[NDTOR_ENTRY:.*]], label %[[NDTOR_DONE:.*]]
//
// LLVM:     [[NDTOR_ENTRY]]:
// LLVM:       store ptr %[[LAST]], ptr %[[DTOR_ITER:.*]]
// LLVM:       br label %[[NDTOR_BODY:.*]]
//
//            --- normal dtor loop condition (back-edge) ---
// LLVM:     [[NDTOR_LOOP_COND:.*]]:
// LLVM:       %[[NDCUR_CHECK:.*]] = load ptr, ptr %[[DTOR_ITER]]
// LLVM:       %[[NDTOR_CONT:.*]] = icmp ne ptr %[[NDCUR_CHECK]], %[[BEGIN]]
// LLVM:       br i1 %[[NDTOR_CONT]], label %[[NDTOR_BODY]], label %[[NDTOR_EXIT:.*]]
//
//            --- normal dtor loop body ---
// LLVM:     [[NDTOR_BODY]]:
// LLVM:       %[[NDCUR:.*]] = load ptr, ptr %[[DTOR_ITER]]
// LLVM:       %[[NDPREV:.*]] = getelementptr %struct.S, ptr %[[NDCUR]], i64 -1
// LLVM:       store ptr %[[NDPREV]], ptr %[[DTOR_ITER]]
// LLVM:       call void @_ZN1SD1Ev(ptr{{.*}} %[[NDPREV]])
// LLVM:       br label %[[NDTOR_LOOP_COND]]
//
// LLVM:     [[NDTOR_EXIT]]:
// LLVM:       br label %[[NDTOR_DONE]]
//
// LLVM:     [[NDTOR_DONE]]:
// LLVM:       call void @llvm.stackrestore.p0
// LLVM:       ret void

// OGCG:     define dso_local void @_Z40test_variable_size_partial_array_cleanupi
// OGCG:     [[ENTRY:.*]]:
// OGCG:       %[[N:.*]] = zext i32 %{{.*}} to i64
// OGCG:       %[[VLA:.*]] = alloca %struct.S, i64 %[[N]]
// OGCG:       %[[ISEMPTY:.*]] = icmp eq i64 %[[N]], 0
// OGCG:       br i1 %[[ISEMPTY]], label %[[CTOR_CONT:.*]], label %[[CTOR_LOOP_ENTRY:.*]]
//
//            --- ctor loop ---
// OGCG:     [[CTOR_LOOP_ENTRY]]:
// OGCG:       %[[CTOR_END:.*]] = getelementptr inbounds %struct.S, ptr %[[VLA]], i64 %[[N]]
// OGCG:       br label %[[CTOR_LOOP:.*]]
//
// OGCG:     [[CTOR_LOOP]]:
// OGCG:       %[[CUR:.*]] = phi ptr [ %[[VLA]], %[[CTOR_LOOP_ENTRY]] ], [ %[[NEXT:.*]], %[[INVOKE_CONT:.*]] ]
// OGCG:       invoke void @_ZN1SC1Ev(ptr{{.*}})
// OGCG:         to label %[[INVOKE_CONT]] unwind label %[[LPAD:.*]]
//
// OGCG:     [[INVOKE_CONT]]:
// OGCG:       %[[NEXT]] = getelementptr inbounds %struct.S, ptr %[[CUR]], i64 1
// OGCG:       %[[DONE:.*]] = icmp eq ptr %[[NEXT]], %[[CTOR_END]]
// OGCG:       br i1 %[[DONE]], label %[[CTOR_CONT]], label %[[CTOR_LOOP]]
//
//            --- normal dtor ---
// OGCG:     [[CTOR_CONT]]:
// OGCG:       %[[DTOR_END:.*]] = getelementptr inbounds %struct.S, ptr %[[VLA]], i64 %[[N]]
// OGCG:       %[[DTOR_ISEMPTY:.*]] = icmp eq ptr %[[VLA]], %[[DTOR_END]]
// OGCG:       br i1 %[[DTOR_ISEMPTY]], label %[[DTOR_DONE:.*]], label %[[DTOR_LOOP:.*]]
//
// OGCG:     [[DTOR_LOOP]]:
// OGCG:       %[[PAST:.*]] = phi ptr [ %[[DTOR_END]], %[[CTOR_CONT]] ], [ %[[PREV:.*]], %[[DTOR_LOOP]] ]
// OGCG:       %[[PREV]] = getelementptr inbounds %struct.S, ptr %[[PAST]], i64 -1
// OGCG:       call void @_ZN1SD1Ev(ptr{{.*}} %[[PREV]])
// OGCG:       %[[DDONE:.*]] = icmp eq ptr %[[PREV]], %[[VLA]]
// OGCG:       br i1 %[[DDONE]], label %[[DTOR_DONE]], label %[[DTOR_LOOP]]
//
// OGCG:     [[DTOR_DONE]]:
// OGCG:       call void @llvm.stackrestore.p0(ptr %{{.*}})
// OGCG:       ret void
//
//            --- landing pad + cleanup guard ---
// OGCG:     [[LPAD]]:
// OGCG:       landingpad { ptr, i32 }
// OGCG:         cleanup
// OGCG:       %[[PISEMPTY:.*]] = icmp eq ptr %[[VLA]], %[[CUR]]
// OGCG:       br i1 %[[PISEMPTY]], label %[[PDTOR_DONE:.*]], label %[[PDTOR_LOOP:.*]]
//
//            --- partial dtor loop ---
// OGCG:     [[PDTOR_LOOP]]:
// OGCG:       %[[PPAST:.*]] = phi ptr [ %[[CUR]], %[[LPAD]] ], [ %[[PPREV:.*]], %[[PDTOR_LOOP]] ]
// OGCG:       %[[PPREV]] = getelementptr inbounds %struct.S, ptr %[[PPAST]], i64 -1
// OGCG:       call void @_ZN1SD1Ev(ptr{{.*}} %[[PPREV]])
// OGCG:       %[[PDDONE:.*]] = icmp eq ptr %[[PPREV]], %[[VLA]]
// OGCG:       br i1 %[[PDDONE]], label %[[PDTOR_DONE]], label %[[PDTOR_LOOP]]
//
// OGCG:     [[PDTOR_DONE]]:
// OGCG:       br label %[[EH_RESUME:.*]]
//
// OGCG:     [[EH_RESUME]]:
// OGCG:       resume { ptr, i32 }

void test_multi_dim_vla(int n, int m) {
    S s[n][m];
}

// CIR-BEFORE-LPP:     cir.func {{.*}} @_Z18test_multi_dim_vlaii
// CIR-BEFORE-LPP:       %[[N_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["n", init]
// CIR-BEFORE-LPP:       %[[M_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["m", init]
// CIR-BEFORE-LPP:       %[[N_VAL:.*]] = cir.load {{.*}} %[[N_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR-BEFORE-LPP:       %[[N:.*]] = cir.cast integral %[[N_VAL]] : !s32i -> !u64i
// CIR-BEFORE-LPP:       %[[M_VAL:.*]] = cir.load {{.*}} %[[M_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR-BEFORE-LPP:       %[[M:.*]] = cir.cast integral %[[M_VAL]] : !s32i -> !u64i
// CIR-BEFORE-LPP:       cir.stacksave
// CIR-BEFORE-LPP:       cir.cleanup.scope {
// CIR-BEFORE-LPP:         %[[NM:.*]] = cir.mul nuw %[[N]], %[[M]] : !u64i
// CIR-BEFORE-LPP:         %[[VLA:.*]] = cir.alloca !rec_S, !cir.ptr<!rec_S>, %[[NM]] : !u64i, ["s", init]
// CIR-BEFORE-LPP:         cir.array.ctor %[[VLA]], {{.*}} : !cir.ptr<!rec_S>, !u64i {
// CIR-BEFORE-LPP:         ^bb0(%[[CTOR_ARG:.*]]: !cir.ptr<!rec_S>):
// CIR-BEFORE-LPP:           cir.call @_ZN1SC1Ev(%[[CTOR_ARG]]) : (!cir.ptr<!rec_S>{{.*}}) -> ()
// CIR-BEFORE-LPP:         } partial_dtor {
// CIR-BEFORE-LPP:         ^bb0(%[[DTOR_ARG:.*]]: !cir.ptr<!rec_S>):
// CIR-BEFORE-LPP:           cir.call @_ZN1SD1Ev(%[[DTOR_ARG]]){{.*}} : (!cir.ptr<!rec_S>{{.*}}) -> ()
// CIR-BEFORE-LPP:         }
// CIR-BEFORE-LPP:         cir.cleanup.scope {
// CIR-BEFORE-LPP:           cir.yield
// CIR-BEFORE-LPP:         } cleanup all {
// CIR-BEFORE-LPP:           cir.array.dtor %[[VLA]], {{.*}} : !cir.ptr<!rec_S>, !u64i {
// CIR-BEFORE-LPP:           ^bb0(%[[DTOR_ARG2:.*]]: !cir.ptr<!rec_S>):
// CIR-BEFORE-LPP:             cir.call @_ZN1SD1Ev(%[[DTOR_ARG2]]){{.*}} : (!cir.ptr<!rec_S>{{.*}}) -> ()
// CIR-BEFORE-LPP:           }
// CIR-BEFORE-LPP:       } cleanup normal {
// CIR-BEFORE-LPP:         cir.stackrestore
// CIR-BEFORE-LPP:       }

// CIR:     cir.func {{.*}} @_Z18test_multi_dim_vlaii
// CIR:       %[[N_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["n", init]
// CIR:       %[[M_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["m", init]
// CIR:       %[[SAVED_STACK:.*]] = cir.alloca !cir.ptr<!u8i>, !cir.ptr<!cir.ptr<!u8i>>, ["saved_stack"]
// CIR:       %[[N_VAL:.*]] = cir.load {{.*}} %[[N_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:       %[[N:.*]] = cir.cast integral %[[N_VAL]] : !s32i -> !u64i
// CIR:       %[[M_VAL:.*]] = cir.load {{.*}} %[[M_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:       %[[M:.*]] = cir.cast integral %[[M_VAL]] : !s32i -> !u64i
// CIR:       cir.cleanup.scope {
// CIR:         %[[NM:.*]] = cir.mul nuw %[[N]], %[[M]] : !u64i
// CIR:         %[[BEGIN:.*]] = cir.alloca !rec_S, !cir.ptr<!rec_S>, %[[NM]] : !u64i, ["s", init]
// CIR:         cir.call @_ZN1SC1Ev
// CIR:         } cleanup eh {
// CIR:           cir.call @_ZN1SD1Ev
// CIR:         } cleanup all {
// CIR:           %[[NM3:.*]] = cir.mul nuw %[[N]], %[[M]] : !u64i
// CIR:           cir.call @_ZN1SD1Ev
// CIR:       } cleanup normal {
// CIR:         cir.stackrestore {{.*}} : !cir.ptr<!u8i>
// CIR:       }

// LLVM:     define dso_local void @_Z18test_multi_dim_vlaii
// LLVM:       %[[N_I32:.*]] = load i32, ptr %{{.*}}
// LLVM:       %[[N:.*]] = sext i32 %[[N_I32]] to i64
// LLVM:       %[[M_I32:.*]] = load i32, ptr %{{.*}}
// LLVM:       %[[M:.*]] = sext i32 %[[M_I32]] to i64
// LLVM:       call ptr @llvm.stacksave.p0()
//
//            --- VLA alloca with n*m ---
// LLVM:       %[[NM:.*]] = mul nuw i64 %[[N]], %[[M]]
// LLVM:       %[[BEGIN:.*]] = alloca %struct.S, i64 %[[NM]]
//
//            --- ctor loop ---
// LLVM:       invoke void @_ZN1SC1Ev(ptr{{.*}})
// LLVM:         to label %{{.*}} unwind label %[[LPAD:.*]]
//
//            --- landing pad + partial dtor ---
// LLVM:     [[LPAD]]:
// LLVM:       landingpad { ptr, i32 }
// LLVM:         cleanup
// LLVM:       call void @_ZN1SD1Ev
// LLVM:       resume { ptr, i32 }
//
//            --- normal dtor ---
// LLVM:       %[[NM2:.*]] = mul nuw i64 %[[N]], %[[M]]
// LLVM:       %[[LAST:.*]] = getelementptr %struct.S, ptr %[[BEGIN]], i64 %[[NM2]]
// LLVM:       %[[DTOR_NE:.*]] = icmp ne ptr %[[LAST]], %[[BEGIN]]
// LLVM:       call void @_ZN1SD1Ev
// LLVM:       call void @llvm.stackrestore.p0
// LLVM:       ret void

// OGCG:     define dso_local void @_Z18test_multi_dim_vlaii
// OGCG:     [[ENTRY:.*]]:
// OGCG:       %[[N:.*]] = zext i32 %{{.*}} to i64
// OGCG:       %[[M:.*]] = zext i32 %{{.*}} to i64
// OGCG:       %[[NM:.*]] = mul nuw i64 %[[N]], %[[M]]
// OGCG:       %[[VLA:.*]] = alloca %struct.S, i64 %[[NM]]
// OGCG:       %[[ISEMPTY:.*]] = icmp eq i64 %{{.*}}, 0
// OGCG:       br i1 %[[ISEMPTY]], label %[[CTOR_CONT:.*]], label %[[CTOR_LOOP_ENTRY:.*]]
//
//            --- ctor loop ---
// OGCG:     [[CTOR_LOOP_ENTRY]]:
// OGCG:       invoke void @_ZN1SC1Ev(ptr{{.*}})
// OGCG:         to label %{{.*}} unwind label %[[LPAD:.*]]
//
//            --- normal dtor ---
// OGCG:     [[CTOR_CONT]]:
// OGCG:       %[[NM2:.*]] = mul nuw i64 %[[N]], %[[M]]
// OGCG:       call void @_ZN1SD1Ev
//
// OGCG:       call void @llvm.stackrestore.p0(ptr %{{.*}})
// OGCG:       ret void
//
//            --- landing pad + partial dtor ---
// OGCG:     [[LPAD]]:
// OGCG:       landingpad { ptr, i32 }
// OGCG:         cleanup
// OGCG:       call void @_ZN1SD1Ev
//
// OGCG:       resume { ptr, i32 }

void test_vla_of_constant_array(int n) {
    S s[n][4];
}

// CIR-BEFORE-LPP:     cir.func {{.*}} @_Z26test_vla_of_constant_arrayi
// CIR-BEFORE-LPP:       %[[N_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["n", init]
// CIR-BEFORE-LPP:       %[[N_VAL:.*]] = cir.load {{.*}} %[[N_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR-BEFORE-LPP:       %[[N:.*]] = cir.cast integral %[[N_VAL]] : !s32i -> !u64i
// CIR-BEFORE-LPP:       cir.stacksave
// CIR-BEFORE-LPP:       cir.cleanup.scope {
// CIR-BEFORE-LPP:         %[[VLA:.*]] = cir.alloca !cir.array<!rec_S x 4>, !cir.ptr<!cir.array<!rec_S x 4>>, %[[N]] : !u64i, ["s", init]
// CIR-BEFORE-LPP:         %[[FOUR:.*]] = cir.const #cir.int<4> : !u64i
// CIR-BEFORE-LPP:         %[[TOTAL:.*]] = cir.mul nuw %[[N]], %[[FOUR]] : !u64i
// CIR-BEFORE-LPP:         %[[ELEM_PTR:.*]] = cir.cast bitcast %[[VLA]] : !cir.ptr<!cir.array<!rec_S x 4>> -> !cir.ptr<!rec_S>
// CIR-BEFORE-LPP:         cir.array.ctor %[[ELEM_PTR]], %[[TOTAL]] : !cir.ptr<!rec_S>, !u64i {
// CIR-BEFORE-LPP:         ^bb0(%[[CTOR_ARG:.*]]: !cir.ptr<!rec_S>):
// CIR-BEFORE-LPP:           cir.call @_ZN1SC1Ev(%[[CTOR_ARG]]) : (!cir.ptr<!rec_S>{{.*}}) -> ()
// CIR-BEFORE-LPP:         } partial_dtor {
// CIR-BEFORE-LPP:         ^bb0(%[[DTOR_ARG:.*]]: !cir.ptr<!rec_S>):
// CIR-BEFORE-LPP:           cir.call @_ZN1SD1Ev(%[[DTOR_ARG]]){{.*}} : (!cir.ptr<!rec_S>{{.*}}) -> ()
// CIR-BEFORE-LPP:         }
// CIR-BEFORE-LPP:         cir.cleanup.scope {
// CIR-BEFORE-LPP:           cir.yield
// CIR-BEFORE-LPP:         } cleanup all {
// CIR-BEFORE-LPP:           %{{.*}} = cir.const #cir.int<4> : !u64i
// CIR-BEFORE-LPP:           %{{.*}} = cir.mul nuw %[[N]], %{{.*}} : !u64i
// CIR-BEFORE-LPP:           %{{.*}} = cir.cast bitcast %[[VLA]] : !cir.ptr<!cir.array<!rec_S x 4>> -> !cir.ptr<!rec_S>
// CIR-BEFORE-LPP:           cir.array.dtor {{.*}} : !cir.ptr<!rec_S>, !u64i {
// CIR-BEFORE-LPP:           ^bb0(%[[DTOR_ARG2:.*]]: !cir.ptr<!rec_S>):
// CIR-BEFORE-LPP:             cir.call @_ZN1SD1Ev(%[[DTOR_ARG2]]){{.*}} : (!cir.ptr<!rec_S>{{.*}}) -> ()
// CIR-BEFORE-LPP:           }
// CIR-BEFORE-LPP:       } cleanup normal {
// CIR-BEFORE-LPP:         cir.stackrestore
// CIR-BEFORE-LPP:       }

// CIR:     cir.func {{.*}} @_Z26test_vla_of_constant_arrayi
// CIR:       %[[N_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["n", init]
// CIR:       %[[N_VAL:.*]] = cir.load {{.*}} %[[N_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:       %[[N:.*]] = cir.cast integral %[[N_VAL]] : !s32i -> !u64i
// CIR:       cir.cleanup.scope {
// CIR:         %[[VLA:.*]] = cir.alloca !cir.array<!rec_S x 4>, !cir.ptr<!cir.array<!rec_S x 4>>, %[[N]] : !u64i, ["s", init]
// CIR:         %[[FOUR:.*]] = cir.const #cir.int<4> : !u64i
// CIR:         %[[TOTAL:.*]] = cir.mul nuw %[[N]], %[[FOUR]] : !u64i
// CIR:         %[[ELEM_PTR:.*]] = cir.cast bitcast %[[VLA]] : !cir.ptr<!cir.array<!rec_S x 4>> -> !cir.ptr<!rec_S>
// CIR:         cir.call @_ZN1SC1Ev
// CIR:         } cleanup eh {
// CIR:           cir.call @_ZN1SD1Ev
// CIR:         } cleanup all {
// CIR:           %{{.*}} = cir.const #cir.int<4> : !u64i
// CIR:           %{{.*}} = cir.mul nuw %[[N]], %{{.*}} : !u64i
// CIR:           %{{.*}} = cir.cast bitcast %[[VLA]] : !cir.ptr<!cir.array<!rec_S x 4>> -> !cir.ptr<!rec_S>
// CIR:           cir.call @_ZN1SD1Ev
// CIR:       } cleanup normal {
// CIR:         cir.stackrestore {{.*}} : !cir.ptr<!u8i>
// CIR:       }

// LLVM:     define dso_local void @_Z26test_vla_of_constant_arrayi
// LLVM:       %[[N_I32:.*]] = load i32, ptr %{{.*}}
// LLVM:       %[[N:.*]] = sext i32 %[[N_I32]] to i64
// LLVM:       call ptr @llvm.stacksave.p0()
//
//            --- VLA alloca with [4 x %struct.S] ---
// LLVM:       %[[BEGIN:.*]] = alloca [4 x %struct.S], i64 %[[N]]
// LLVM:       %[[TOTAL:.*]] = mul nuw i64 %[[N]], 4
//
//            --- ctor loop ---
// LLVM:       invoke void @_ZN1SC1Ev(ptr{{.*}})
// LLVM:         to label %{{.*}} unwind label %[[LPAD:.*]]
//
//            --- landing pad + partial dtor ---
// LLVM:     [[LPAD]]:
// LLVM:       landingpad { ptr, i32 }
// LLVM:         cleanup
// LLVM:       call void @_ZN1SD1Ev
// LLVM:       resume { ptr, i32 }
//
//            --- normal dtor ---
// LLVM:       %[[TOTAL2:.*]] = mul nuw i64 %[[N]], 4
// LLVM:       call void @_ZN1SD1Ev
// LLVM:       call void @llvm.stackrestore.p0
// LLVM:       ret void

// OGCG:     define dso_local void @_Z26test_vla_of_constant_arrayi
// OGCG:     [[ENTRY:.*]]:
// OGCG:       %[[N:.*]] = zext i32 %{{.*}} to i64
// OGCG:       %[[VLA:.*]] = alloca [4 x %struct.S], i64 %[[N]]
// OGCG:       %[[ARRAY_BEGIN:.*]] = getelementptr inbounds [4 x %struct.S], ptr %[[VLA]], i32 0, i32 0
// OGCG:       %[[TOTAL:.*]] = mul nuw i64 %[[N]], 4
// OGCG:       %[[ISEMPTY:.*]] = icmp eq i64 %[[TOTAL]], 0
// OGCG:       br i1 %[[ISEMPTY]], label %[[CTOR_CONT:.*]], label %[[CTOR_LOOP_ENTRY:.*]]
//
//            --- ctor loop ---
// OGCG:     [[CTOR_LOOP_ENTRY]]:
// OGCG:       invoke void @_ZN1SC1Ev(ptr{{.*}})
// OGCG:         to label %{{.*}} unwind label %[[LPAD:.*]]
//
//            --- normal dtor ---
// OGCG:     [[CTOR_CONT]]:
// OGCG:       %[[TOTAL2:.*]] = mul nuw i64 %[[N]], 4
// OGCG:       call void @_ZN1SD1Ev
//
// OGCG:       call void @llvm.stackrestore.p0(ptr %{{.*}})
// OGCG:       ret void
//
//            --- landing pad + partial dtor ---
// OGCG:     [[LPAD]]:
// OGCG:       landingpad { ptr, i32 }
// OGCG:         cleanup
// OGCG:       call void @_ZN1SD1Ev
//
// OGCG:       resume { ptr, i32 }
