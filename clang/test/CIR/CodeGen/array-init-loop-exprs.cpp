// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

// Non-trivially copyable element type.
struct NonTrivial {
    NonTrivial(const NonTrivial &);
    int val;
};

// A struct with a non-trivially copyable array member. The implicit copy
// constructor will use ArrayInitLoopExpr to copy each element.
struct HasNonTrivialArray {
    NonTrivial arr[3];
};

// CIR-LABEL: cir.func no_inline comdat linkonce_odr @_ZN18HasNonTrivialArrayC2ERKS_({{.*}}) special_member<#cir.cxx_ctor<!rec_HasNonTrivialArray, copy>> 
// CIR: %[[THIS_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_HasNonTrivialArray>, !cir.ptr<!cir.ptr<!rec_HasNonTrivialArray>>, ["this", init]
// CIR: %[[RHS_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_HasNonTrivialArray>, !cir.ptr<!cir.ptr<!rec_HasNonTrivialArray>>, ["", init, const]
// CIR: %[[ITR_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_NonTrivial>, !cir.ptr<!cir.ptr<!rec_NonTrivial>>, ["arrayinit.temp"]
// CIR: %[[THIS_LOAD:.*]] = cir.load %[[THIS_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_HasNonTrivialArray>>, !cir.ptr<!rec_HasNonTrivialArray>
// CIR: %[[THIS_ARR:.*]] = cir.get_member %[[THIS_LOAD]][0] {name = "arr"} : !cir.ptr<!rec_HasNonTrivialArray> -> !cir.ptr<!cir.array<!rec_NonTrivial x 3>>
// CIR: %[[RHS_LOAD:.*]] = cir.load %[[RHS_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_HasNonTrivialArray>>, !cir.ptr<!rec_HasNonTrivialArray>
// CIR: %[[RHS_ARR:.*]] = cir.get_member %[[RHS_LOAD]][0] {name = "arr"} : !cir.ptr<!rec_HasNonTrivialArray> -> !cir.ptr<!cir.array<!rec_NonTrivial x 3>>
// CIR: %[[THIS_ARR_DECAY:.*]] = cir.cast array_to_ptrdecay %[[THIS_ARR]] : !cir.ptr<!cir.array<!rec_NonTrivial x 3>> -> !cir.ptr<!rec_NonTrivial>
// CIR: cir.store {{.*}}%[[THIS_ARR_DECAY]], %[[ITR_ALLOCA]] : !cir.ptr<!rec_NonTrivial>, !cir.ptr<!cir.ptr<!rec_NonTrivial>>
// CIR: %[[SIZE_CONST:.*]] = cir.const #cir.int<3>
// CIR: %[[END_ITR:.*]] = cir.ptr_stride %[[THIS_ARR_DECAY]], %[[SIZE_CONST]] : (!cir.ptr<!rec_NonTrivial>, !s64i) -> !cir.ptr<!rec_NonTrivial>
// CIR: cir.do {
// CIR:   %[[ITR_LOAD:.*]] = cir.load {{.*}}%[[ITR_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_NonTrivial>>, !cir.ptr<!rec_NonTrivial>
// CIR:   %[[IDX:.*]] = cir.ptr_diff %[[ITR_LOAD]], %[[THIS_ARR_DECAY]] : !cir.ptr<!rec_NonTrivial> -> !s64i
// CIR:   %[[RHS_ELT:.*]] = cir.get_element %[[RHS_ARR]][%[[IDX]] : !s64i] : !cir.ptr<!cir.array<!rec_NonTrivial x 3>> -> !cir.ptr<!rec_NonTrivial>
// CIR:   cir.call @_ZN10NonTrivialC1ERKS_(%[[ITR_LOAD]], %[[RHS_ELT]]) : (!cir.ptr<!rec_NonTrivial> {{.*}}, !cir.ptr<!rec_NonTrivial> {{.*}}) -> ()
// CIR:   %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CIR:   %[[NEXT_ITR:.*]] = cir.ptr_stride %[[ITR_LOAD]], %[[ONE]] : (!cir.ptr<!rec_NonTrivial>, !s64i) -> !cir.ptr<!rec_NonTrivial>
// CIR:   cir.store align(8) %[[NEXT_ITR]], %[[ITR_ALLOCA]] : !cir.ptr<!rec_NonTrivial>, !cir.ptr<!cir.ptr<!rec_NonTrivial>>
// CIR:   cir.yield
// CIR: } while {
// CIR:   %[[ITR_LOAD:.*]] = cir.load {{.*}}%[[ITR_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_NonTrivial>>, !cir.ptr<!rec_NonTrivial>
// CIR:   %[[COND:.*]] = cir.cmp ne %[[ITR_LOAD]], %[[END_ITR]] : !cir.ptr<!rec_NonTrivial>
// CIR:   cir.condition(%[[COND]])
// CIR: }
// CIR: cir.return

// CIR-LABEL: cir.func {{.*}}@_ZN18HasNonTrivialArrayC1ERKS_(
// CIR: cir.call @_ZN18HasNonTrivialArrayC2ERKS_(
// CIR-LABEL: cir.func{{.*}}make_copy(
// CIR: cir.call @_ZN18HasNonTrivialArrayC1ERKS_(


// LLVM-LABEL: define {{.*}}@_ZN18HasNonTrivialArrayC2ERKS_(
// LLVM: %[[THIS_ALLOCA:.*]] = alloca ptr
// LLVM: %[[RHS_ALLOCA:.*]] = alloca ptr
// LLVM: %[[ITR_ALLOCA:.*]] = alloca ptr
// LLVM: %[[THIS_LOAD:.*]] = load ptr, ptr %[[THIS_ALLOCA]]
// LLVM: %[[THIS_ARR:.*]] = getelementptr inbounds nuw %struct.HasNonTrivialArray, ptr %[[THIS_LOAD]], i32 0, i32 0
// LLVM: %[[RHS_LOAD:.*]] = load ptr, ptr %[[RHS_ALLOCA]]
// LLVM: %[[RHS_ARR:.*]] = getelementptr inbounds nuw %struct.HasNonTrivialArray, ptr %[[RHS_LOAD]], i32 0, i32 0
// LLVM: %[[THIS_ARR_DECAY:.*]] = getelementptr %struct.NonTrivial, ptr %[[THIS_ARR]], i32 0
// LLVM: store ptr %[[THIS_ARR_DECAY]], ptr %[[ITR_ALLOCA]]
// LLVM: %[[END_ITR:.*]] = getelementptr %struct.NonTrivial, ptr %[[THIS_ARR_DECAY]], i64 3
// LLVM: br label %[[BODY:.*]]

// LLVM: [[COND_BLOCK:.*]]:
// LLVM: %[[LOAD_ITR:.*]] = load ptr, ptr %[[ITR_ALLOCA]]
// LLVM: %[[COND:.*]] = icmp ne ptr %[[LOAD_ITR]], %[[END_ITR]]
// LLVM: br i1 %[[COND]], label %[[BODY]], label %[[ENDBLOCK:.*]]

// LLVM: [[BODY]]:
// LLVM: %[[LOAD_ITR:.*]] = load ptr, ptr %[[ITR_ALLOCA]]
// LLVM: %[[ITR_PTR:.*]] = ptrtoint ptr %[[LOAD_ITR]] to i64
// LLVM: %[[START_PTR:.*]] = ptrtoint ptr %[[THIS_ARR_DECAY]] to i64
// LLVM: %[[PTR_DIFF:.*]] = sub i64 %[[ITR_PTR]], %[[START_PTR]]
// LLVM: %[[IDX:.*]] = sdiv exact i64 %[[PTR_DIFF]], 4
// LLVM: %[[RHS_ELT:.*]] = getelementptr [3 x %struct.NonTrivial], ptr %[[RHS_ARR]], i32 0, i64 %[[IDX]]
// LLVM: call void @_ZN10NonTrivialC1ERKS_(ptr {{.*}}%[[LOAD_ITR]], ptr {{.*}}%[[RHS_ELT]])
// LLVM: %[[NEXT_ITR:.*]] = getelementptr %struct.NonTrivial, ptr %[[LOAD_ITR]], i64 1
// LLVM: store ptr %[[NEXT_ITR]], ptr %[[ITR_ALLOCA]]
// LLVM: br label %[[COND_BLOCK]]

// LLVM: [[ENDBLOCK]]: 
// LLVM: ret void

// LLVM-LABEL: define {{.*}}@_ZN18HasNonTrivialArrayC1ERKS_(
// LLVM: call void @_ZN18HasNonTrivialArrayC2ERKS_(
// LLVM-LABEL: define dso_local %struct.HasNonTrivialArray @make_copy(
// LLVM: call void @_ZN18HasNonTrivialArrayC1ERKS_(

// OGCG-LABEL: define {{.*}}@make_copy(
// OGCG: call void @_ZN18HasNonTrivialArrayC1ERKS_(
//
// OGCG-LABEL: define {{.*}}@_ZN18HasNonTrivialArrayC1ERKS_(
// OGCG: call void @_ZN18HasNonTrivialArrayC2ERKS_(
//
// Note: CIR lowering puts these in a different order, Classic codegen seems to
// emit these early and the bases later, so these two declarations are out of
// order.
// OGCG-LABEL: define {{.*}}@make_multi_copy(
// OGCG: call void @_ZN16HasMultiDimArrayC1ERKS_(

// OGCG-LABEL: define {{.*}}@_ZN16HasMultiDimArrayC1ERKS_(
// OGCG: call void @_ZN16HasMultiDimArrayC2ERKS_(

// OGCG-LABEL: define {{.*}}@_ZN18HasNonTrivialArrayC2ERKS_(
// OGCG: %[[THIS_ALLOCA:.*]] = alloca ptr
// OGCG: %[[RHS_ALLOCA:.*]] = alloca ptr
// OGCG: %[[THIS_LOAD:.*]] = load ptr, ptr %[[THIS_ALLOCA]]
// OGCG: %[[THIS_ARR:.*]] = getelementptr inbounds nuw %struct.HasNonTrivialArray, ptr %[[THIS_LOAD]], i32 0, i32 0
// OGCG: %[[RHS_LOAD:.*]] = load ptr, ptr %[[RHS_ALLOCA]]
// OGCG: %[[RHS_ARR:.*]] = getelementptr inbounds nuw %struct.HasNonTrivialArray, ptr %[[RHS_LOAD]], i32 0, i32 0
// OGCG: %[[ITR_BEGIN:.*]] = getelementptr inbounds [3 x %struct.NonTrivial], ptr %[[THIS_ARR]], i64 0, i64 0
// OGCG: br label %[[ARR_BODY:.*]]

// OGCG: [[ARR_BODY]]:
// OGCG: %[[IDX:.*]] = phi i64 [ 0, %entry ], [ %[[ITR_NEXT:.*]], %[[ARR_BODY]] ]
// OGCG: %[[ITR:.*]] = getelementptr inbounds %struct.NonTrivial, ptr %[[ITR_BEGIN]], i64 %[[IDX]]
// OGCG: %[[RHS_ITR:.*]] = getelementptr inbounds nuw [3 x %struct.NonTrivial], ptr %[[RHS_ARR]], i64 0, i64 %[[IDX]]
// OGCG: call void @_ZN10NonTrivialC1ERKS_(ptr {{.*}}%[[ITR]], ptr {{.*}}%[[RHS_ITR]])
// OGCG: %[[ITR_NEXT]] = add nuw i64 %[[IDX]], 1
// OGCG: %[[COND:.*]] = icmp eq i64 %[[ITR_NEXT]], 3
// OGCG: br i1 %[[COND]], label %[[END_BLOCK:.*]], label %[[ARR_BODY]]

// OGCG: [[END_BLOCK]]:
// OGCG:   ret void

extern "C" HasNonTrivialArray make_copy(const HasNonTrivialArray &src) {
    return src;
}

// Multi-dimensional: the outer loop iterates over rows, the inner loop
// (a nested ArrayInitLoopExpr) iterates over columns.
struct HasMultiDimArray {
    NonTrivial arr[2][3][4];
};

// CIR-LABEL: cir.func {{.*}}@_ZN16HasMultiDimArrayC2ERKS_({{.*}}) special_member<#cir.cxx_ctor<!rec_HasMultiDimArray, copy>> 
// CIR: %[[THIS_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_HasMultiDimArray>, !cir.ptr<!cir.ptr<!rec_HasMultiDimArray>>, ["this", init]
// CIR: %[[RHS_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_HasMultiDimArray>, !cir.ptr<!cir.ptr<!rec_HasMultiDimArray>>, ["", init, const]
// CIR: %[[ITR1_ALLOCA:.*]] = cir.alloca !cir.ptr<!cir.array<!cir.array<!rec_NonTrivial x 4> x 3>>, !cir.ptr<!cir.ptr<!cir.array<!cir.array<!rec_NonTrivial x 4> x 3>>>, ["arrayinit.temp"]
// CIR: %[[ITR2_ALLOCA:.*]] = cir.alloca !cir.ptr<!cir.array<!rec_NonTrivial x 4>>, !cir.ptr<!cir.ptr<!cir.array<!rec_NonTrivial x 4>>>, ["arrayinit.temp"]
// CIR: %[[ITR3_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_NonTrivial>, !cir.ptr<!cir.ptr<!rec_NonTrivial>>, ["arrayinit.temp"] {alignment = 8 : i64}
// CIR: %[[THIS_LOAD:.*]] = cir.load %[[THIS_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_HasMultiDimArray>>, !cir.ptr<!rec_HasMultiDimArray>
// CIR: %[[THIS_ARR:.*]] = cir.get_member %[[THIS_LOAD]][0] {name = "arr"} : !cir.ptr<!rec_HasMultiDimArray> -> !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_NonTrivial x 4> x 3> x 2>>
// CIR: %[[RHS_LOAD:.*]] = cir.load %[[RHS_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_HasMultiDimArray>>, !cir.ptr<!rec_HasMultiDimArray>
// CIR: %[[RHS_ARR:.*]] = cir.get_member %[[RHS_LOAD]][0] {name = "arr"} : !cir.ptr<!rec_HasMultiDimArray> -> !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_NonTrivial x 4> x 3> x 2>>
// CIR: %[[THIS_ARR_DECAY:.*]] = cir.cast array_to_ptrdecay %[[THIS_ARR]] : !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_NonTrivial x 4> x 3> x 2>> -> !cir.ptr<!cir.array<!cir.array<!rec_NonTrivial x 4> x 3>>
// CIR: cir.store {{.*}}%[[THIS_ARR_DECAY]], %[[ITR1_ALLOCA]] : !cir.ptr<!cir.array<!cir.array<!rec_NonTrivial x 4> x 3>>, !cir.ptr<!cir.ptr<!cir.array<!cir.array<!rec_NonTrivial x 4> x 3>>>
// CIR: %[[SIZE1_CONST:.*]] = cir.const #cir.int<2> : !s64i
// CIR: %[[END_ITR1:.*]] = cir.ptr_stride %[[THIS_ARR_DECAY]], %[[SIZE1_CONST]] : (!cir.ptr<!cir.array<!cir.array<!rec_NonTrivial x 4> x 3>>, !s64i) -> !cir.ptr<!cir.array<!cir.array<!rec_NonTrivial x 4> x 3>>
// CIR: cir.do {
// CIR:   %[[ITR1_LOAD:.*]] = cir.load {{.*}}%[[ITR1_ALLOCA]] : !cir.ptr<!cir.ptr<!cir.array<!cir.array<!rec_NonTrivial x 4> x 3>>>, !cir.ptr<!cir.array<!cir.array<!rec_NonTrivial x 4> x 3>>
// CIR:   %[[IDX1:.*]] = cir.ptr_diff %[[ITR1_LOAD]], %[[THIS_ARR_DECAY]] : !cir.ptr<!cir.array<!cir.array<!rec_NonTrivial x 4> x 3>> -> !s64i
// CIR:   %[[RHS_ELT1:.*]] = cir.get_element %[[RHS_ARR]][%[[IDX1]] : !s64i] : !cir.ptr<!cir.array<!cir.array<!cir.array<!rec_NonTrivial x 4> x 3> x 2>> -> !cir.ptr<!cir.array<!cir.array<!rec_NonTrivial x 4> x 3>>
// CIR:   %[[ARR1_DECAY:.*]] = cir.cast array_to_ptrdecay %[[ITR1_LOAD]] : !cir.ptr<!cir.array<!cir.array<!rec_NonTrivial x 4> x 3>> -> !cir.ptr<!cir.array<!rec_NonTrivial x 4>>
// CIR:   cir.store {{.*}}%[[ARR1_DECAY]], %[[ITR2_ALLOCA]] : !cir.ptr<!cir.array<!rec_NonTrivial x 4>>, !cir.ptr<!cir.ptr<!cir.array<!rec_NonTrivial x 4>>>
// CIR:   %[[SIZE2_CONST:.*]] = cir.const #cir.int<3> : !s64i
// CIR:   %[[END_ITR2:.*]] = cir.ptr_stride %[[ARR1_DECAY]], %[[SIZE2_CONST]] : (!cir.ptr<!cir.array<!rec_NonTrivial x 4>>, !s64i) -> !cir.ptr<!cir.array<!rec_NonTrivial x 4>>
// CIR:   cir.do {
// CIR:     %[[ITR2_LOAD:.*]] = cir.load {{.*}}%[[ITR2_ALLOCA]] : !cir.ptr<!cir.ptr<!cir.array<!rec_NonTrivial x 4>>>, !cir.ptr<!cir.array<!rec_NonTrivial x 4>>
// CIR:     %[[IDX2:.*]] = cir.ptr_diff %[[ITR2_LOAD]], %[[ARR1_DECAY]] : !cir.ptr<!cir.array<!rec_NonTrivial x 4>> -> !s64i
// CIR:     %[[RHS_ELT2:.*]] = cir.get_element %[[RHS_ELT1]][%[[IDX2]] : !s64i] : !cir.ptr<!cir.array<!cir.array<!rec_NonTrivial x 4> x 3>> -> !cir.ptr<!cir.array<!rec_NonTrivial x 4>>
// CIR:     %[[ARR2_DECAY:.*]] = cir.cast array_to_ptrdecay %[[ITR2_LOAD]] : !cir.ptr<!cir.array<!rec_NonTrivial x 4>> -> !cir.ptr<!rec_NonTrivial>
// CIR:     cir.store {{.*}}%[[ARR2_DECAY]], %[[ITR3_ALLOCA]] : !cir.ptr<!rec_NonTrivial>, !cir.ptr<!cir.ptr<!rec_NonTrivial>>
// CIR:     %[[SIZE3_CONST:.*]] = cir.const #cir.int<4> : !s64i
// CIR:     %[[END_ITR3:.*]] = cir.ptr_stride %[[ARR2_DECAY]], %[[SIZE3_CONST]] : (!cir.ptr<!rec_NonTrivial>, !s64i) -> !cir.ptr<!rec_NonTrivial>
// CIR:     cir.do {
// CIR:       %[[ITR3_LOAD:.*]] = cir.load {{.*}}%[[ITR3_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_NonTrivial>>, !cir.ptr<!rec_NonTrivial>
// CIR:       %[[IDX3:.*]] = cir.ptr_diff %[[ITR3_LOAD]], %[[ARR2_DECAY]] : !cir.ptr<!rec_NonTrivial> -> !s64i
// CIR:       %[[RHS_ELT3:.*]] = cir.get_element %[[RHS_ELT2]][%[[IDX3]] : !s64i] : !cir.ptr<!cir.array<!rec_NonTrivial x 4>> -> !cir.ptr<!rec_NonTrivial>
// CIR:       cir.call @_ZN10NonTrivialC1ERKS_(%[[ITR3_LOAD]], %[[RHS_ELT3]]) : (!cir.ptr<!rec_NonTrivial> {{.*}}, !cir.ptr<!rec_NonTrivial> {{.*}}) -> ()
// CIR:       %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CIR:       %[[NEXT_ITR3:.*]] = cir.ptr_stride %[[ITR3_LOAD]], %[[ONE]] : (!cir.ptr<!rec_NonTrivial>, !s64i) -> !cir.ptr<!rec_NonTrivial>
// CIR:       cir.store {{.*}}%[[NEXT_ITR3]], %[[ITR3_ALLOCA]] : !cir.ptr<!rec_NonTrivial>, !cir.ptr<!cir.ptr<!rec_NonTrivial>>
// CIR:       cir.yield
// CIR:     } while {
// CIR:     %[[ITR3_LOAD:.*]] = cir.load {{.*}}%[[ITR3_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_NonTrivial>>, !cir.ptr<!rec_NonTrivial>
// CIR:     %[[COND3:.*]] = cir.cmp ne %[[ITR3_LOAD]], %[[END_ITR3]] : !cir.ptr<!rec_NonTrivial>
// CIR:     cir.condition(%[[COND3]])
// CIR:     }
// CIR:     %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CIR:     %[[NEXT_ITR2:.*]] = cir.ptr_stride %[[ITR2_LOAD]], %[[ONE]] : (!cir.ptr<!cir.array<!rec_NonTrivial x 4>>, !s64i) -> !cir.ptr<!cir.array<!rec_NonTrivial x 4>>
// CIR:     cir.store {{.*}}%[[NEXT_ITR2]], %[[ITR2_ALLOCA]] : !cir.ptr<!cir.array<!rec_NonTrivial x 4>>, !cir.ptr<!cir.ptr<!cir.array<!rec_NonTrivial x 4>>>
// CIR:     cir.yield
// CIR:   } while {
// CIR:   %[[ITR2_LOAD:.*]] = cir.load align(8) %[[ITR2_ALLOCA]] : !cir.ptr<!cir.ptr<!cir.array<!rec_NonTrivial x 4>>>, !cir.ptr<!cir.array<!rec_NonTrivial x 4>>
// CIR:   %[[COND2:.*]] = cir.cmp ne %[[ITR2_LOAD]], %[[END_ITR2]] : !cir.ptr<!cir.array<!rec_NonTrivial x 4>>
// CIR:   cir.condition(%[[COND2]])
// CIR:   }
// CIR:   %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CIR:   %[[NEXT_ITR1:.*]] = cir.ptr_stride %[[ITR1_LOAD]], %[[ONE]] : (!cir.ptr<!cir.array<!cir.array<!rec_NonTrivial x 4> x 3>>, !s64i) -> !cir.ptr<!cir.array<!cir.array<!rec_NonTrivial x 4> x 3>>
// CIR:   cir.store {{.*}}%[[NEXT_ITR1]], %[[ITR1_ALLOCA]] : !cir.ptr<!cir.array<!cir.array<!rec_NonTrivial x 4> x 3>>, !cir.ptr<!cir.ptr<!cir.array<!cir.array<!rec_NonTrivial x 4> x 3>>>
// CIR:   cir.yield
// CIR: } while {
// CIR: %[[ITR1_LOAD:.*]] = cir.load {{.*}}%[[ITR1_ALLOCA]] : !cir.ptr<!cir.ptr<!cir.array<!cir.array<!rec_NonTrivial x 4> x 3>>>, !cir.ptr<!cir.array<!cir.array<!rec_NonTrivial x 4> x 3>>
// CIR: %[[COND1:.*]] = cir.cmp ne %[[ITR1_LOAD]], %[[END_ITR1]] : !cir.ptr<!cir.array<!cir.array<!rec_NonTrivial x 4> x 3>>
// CIR: cir.condition(%[[COND1]])
// CIR: }
// CIR: cir.return

// CIR-LABEL: cir.func {{.*}}@_ZN16HasMultiDimArrayC1ERKS_(
// CIR:    cir.call @_ZN16HasMultiDimArrayC2ERKS_(

// CIR-LABEL: cir.func{{.*}}make_multi_copy(
// CIR: cir.call @_ZN16HasMultiDimArrayC1ERKS_(

// LLVM-LABEL: define linkonce_odr void @_ZN16HasMultiDimArrayC2ERKS_(
// LLVM: %[[THIS_ALLOCA:.*]] = alloca ptr
// LLVM: %[[RHS_ALLOCA:.*]] = alloca ptr
// LLVM: %[[ITR1_ALLOCA:.*]] = alloca ptr
// LLVM: %[[ITR2_ALLOCA:.*]] = alloca ptr
// LLVM: %[[ITR3_ALLOCA:.*]] = alloca ptr
// LLVM: %[[THIS_LOAD:.*]] = load ptr, ptr %[[THIS_ALLOCA]]
// LLVM: %[[THIS_ARR:.*]] = getelementptr inbounds nuw %struct.HasMultiDimArray, ptr %[[THIS_LOAD]], i32 0, i32 0
// LLVM: %[[RHS_LOAD:.*]] = load ptr, ptr %[[RHS_ALLOCA]]
// LLVM: %[[RHS_ARR:.*]] = getelementptr inbounds nuw %struct.HasMultiDimArray, ptr %[[RHS_LOAD]], i32 0, i32 0
// LLVM: %[[THIS_ARR_DECAY:.*]] = getelementptr [3 x [4 x %struct.NonTrivial]], ptr %[[THIS_ARR]], i32 0
// LLVM: store ptr %[[THIS_ARR_DECAY]], ptr %[[ITR1_ALLOCA]]
// LLVM: %[[END_ITR1:.*]] = getelementptr [3 x [4 x %struct.NonTrivial]], ptr %[[THIS_ARR_DECAY]], i64 2
// LLVM: br label %[[WHILE1_BODY:.*]]

// LLVM: [[WHILE1_COND:.*]]:
// LLVM: %[[ITR1_LOAD:.*]] = load ptr, ptr %[[ITR1_ALLOCA]]
// LLVM: %[[COND1:.*]] = icmp ne ptr %[[ITR1_LOAD]], %[[END_ITR1]]
// LLVM: br i1 %[[COND1]], label %[[WHILE1_BODY]], label %[[END_BLOCK:.*]]

// LLVM: [[WHILE1_BODY]]:
// LLVM: %[[ITR1_LOAD:.*]] = load ptr, ptr %[[ITR1_ALLOCA]]
// LLVM: %[[ITR1_PTR:.*]] = ptrtoint ptr %[[ITR1_LOAD]] to i64
// LLVM: %[[ITR1_START_PTR:.*]] = ptrtoint ptr %[[THIS_ARR_DECAY]] to i64
// LLVM: %[[PTR_DIFF:.*]] = sub i64 %[[ITR1_PTR]], %[[ITR1_START_PTR]]
// LLVM: %[[IDX1:.*]] = sdiv exact i64 %[[PTR_DIFF]], 48
// LLVM: %[[RHS_ELT1:.*]] = getelementptr [2 x [3 x [4 x %struct.NonTrivial]]], ptr %[[RHS_ARR]], i32 0, i64 %[[IDX1]]
// LLVM: %[[ARR1_DECAY:.*]] = getelementptr [4 x %struct.NonTrivial], ptr %[[ITR1_LOAD]], i32 0
// LLVM: store ptr %[[ARR1_DECAY]], ptr %[[ITR2_ALLOCA]]
// LLVM: %[[END_ITR2:.*]] = getelementptr [4 x %struct.NonTrivial], ptr %[[ARR1_DECAY]], i64 3
// LLVM: br label %[[WHILE2_BODY:.*]]

// LLVM: [[WHILE2_COND:.*]]:
// LLVM: %[[ITR2_LOAD:.*]] = load ptr, ptr %[[ITR2_ALLOCA]]
// LLVM: %[[COND2:.*]] = icmp ne ptr %[[ITR2_LOAD]], %[[END_ITR2]]
// LLVM: br i1 %[[COND2]], label %[[WHILE2_BODY]], label %[[WHILE1_BODY_REST:.*]]

// LLVM: [[WHILE2_BODY]]:
// LLVM: %[[ITR2_LOAD:.*]] = load ptr, ptr %[[ITR2_ALLOCA]]
// LLVM: %[[ITR2_PTR:.*]] = ptrtoint ptr %[[ITR2_LOAD]] to i64
// LLVM: %[[ITR2_START_PTR:.*]] = ptrtoint ptr %[[ARR1_DECAY]] to i64
// LLVM: %[[PTR_DIFF:.*]] = sub i64 %[[ITR2_PTR]], %[[ITR2_START_PTR]]
// LLVM: %[[IDX2:.*]] = sdiv exact i64 %[[PTR_DIFF]], 16
// LLVM: %[[RHS_ELT2:.*]] = getelementptr [3 x [4 x %struct.NonTrivial]], ptr %[[RHS_ELT1]], i32 0, i64 %[[IDX2]]
// LLVM: %[[ARR2_DECAY:.*]] = getelementptr %struct.NonTrivial, ptr %[[ITR2_LOAD]], i32 0
// LLVM: store ptr %[[ARR2_DECAY]], ptr %[[ITR3_ALLOCA]]
// LLVM: %[[END_ITR3:.*]] = getelementptr %struct.NonTrivial, ptr %[[ARR2_DECAY]], i64 4
// LLVM: br label %[[WHILE3_BODY:.*]]

// LLVM: [[WHILE3_COND:.*]]:
// LLVM: %[[ITR3_LOAD:.*]] = load ptr, ptr %[[ITR3_ALLOCA]]
// LLVM: %[[COND3:.*]] = icmp ne ptr %[[ITR3_LOAD]], %[[END_ITR3]]
// LLVM: br i1 %[[COND3]], label %[[WHILE3_BODY:.*]], label %[[WHILE2_BODY_REST:.*]]

// LLVM: [[WHILE3_BODY]]:
// LLVM: %[[ITR3_LOAD:.*]] = load ptr, ptr %[[ITR3_ALLOCA]]
// LLVM: %[[ITR3_PTR:.*]] = ptrtoint ptr %[[ITR3_LOAD]] to i64
// LLVM: %[[ITR3_START_PTR:.*]] = ptrtoint ptr %[[ARR2_DECAY]] to i64
// LLVM: %[[PTR_DIFF:.*]] = sub i64 %[[ITR3_PTR]], %[[ITR3_START_PTR]]
// LLVM: %[[IDX3:.*]] = sdiv exact i64 %[[PTR_DIFF]], 4
// LLVM: %[[RHS_ELT3:.*]] = getelementptr [4 x %struct.NonTrivial], ptr %[[RHS_ELT2]], i32 0, i64 %[[IDX3]]
// LLVM: call void @_ZN10NonTrivialC1ERKS_(ptr{{.*}} %[[ITR3_LOAD]], ptr{{.*}} %[[RHS_ELT3]])
// LLVM: %[[NEXT_ITR3:.*]] = getelementptr %struct.NonTrivial, ptr %[[ITR3_LOAD]], i64 1
// LLVM: store ptr %[[NEXT_ITR3]], ptr %[[ITR3_ALLOCA]]
// LLVM: br label %[[WHILE3_COND]]

// LLVM: [[WHILE2_BODY_REST]]:
// LLVM: %[[NEXT_ITR2:.*]] = getelementptr [4 x %struct.NonTrivial], ptr %[[ITR2_LOAD]], i64 1
// LLVM: store ptr %[[NEXT_ITR2]], ptr %[[ITR2_ALLOCA]]
// LLVM: br label %[[WHILE2_COND]]

// LLVM: [[WHILE1_BODY_REST]]:
// LLVM: %[[NEXT_ITR1:.*]] = getelementptr [3 x [4 x %struct.NonTrivial]], ptr %[[ITR1_LOAD]], i64 1
// LLVM: store ptr %[[NEXT_ITR1]], ptr %[[ITR1_ALLOCA]]
// LLVM: br label %[[WHILE1_COND]]

// LLVM: [[END_BLOCK]]:
// LLVM:   ret void

// LLVM-LABEL: define {{.*}}@_ZN16HasMultiDimArrayC1ERKS_(
// LLVM: call void @_ZN16HasMultiDimArrayC2ERKS_(
// LLVM-LABEL: define {{.*}}@make_multi_copy(
// LLVM: call void @_ZN16HasMultiDimArrayC1ERKS_(

// OGCG-LABEL: define linkonce_odr void @_ZN16HasMultiDimArrayC2ERKS_(
// OGCG: %[[THIS_ALLOCA:.*]] = alloca ptr
// OGCG: %[[RHS_ALLOCA:.*]] = alloca ptr
// OGCG: %[[THIS_LOAD:.*]] = load ptr, ptr %[[THIS_ALLOCA]]
// OGCG: %[[THIS_ARR:.*]] = getelementptr inbounds nuw %struct.HasMultiDimArray, ptr %[[THIS_LOAD]], i32 0, i32 0
// OGCG: %[[RHS_LOAD:.*]] = load ptr, ptr %[[RHS_ALLOCA]]
// OGCG: %[[RHS_ARR:.*]] = getelementptr inbounds nuw %struct.HasMultiDimArray, ptr %[[RHS_LOAD]], i32 0, i32 0
// OGCG: %[[ITR1_BEGIN:.*]] = getelementptr inbounds [2 x [3 x [4 x %struct.NonTrivial]]], ptr %[[THIS_ARR]], i64 0, i64 0
// OGCG: br label %[[ARR_BODY1:.*]]

// OGCG: [[ARR_BODY1]]:
// OGCG: %[[IDX1:.*]] = phi i64 [ 0, %entry ], [ %[[ITR1_NEXT:.*]], %[[ARR_BODY1_CTD:.*]] ]
// OGCG: %[[ITR1:.*]] = getelementptr inbounds [3 x [4 x %struct.NonTrivial]], ptr %[[ITR1_BEGIN]], i64 %[[IDX1]]
// OGCG: %[[RHS_ITR1:.*]] = getelementptr inbounds nuw [2 x [3 x [4 x %struct.NonTrivial]]], ptr %[[RHS_ARR]], i64 0, i64 %[[IDX1]]
// OGCG: %[[ITR2_BEGIN:.*]] = getelementptr inbounds [3 x [4 x %struct.NonTrivial]], ptr %[[ITR1]], i64 0, i64 0
// OGCG: br label %[[ARR_BODY2:.*]]

// OGCG: [[ARR_BODY2]]:
// OGCG: %[[IDX2:.*]] = phi i64 [ 0, %[[ARR_BODY1]] ], [ %[[ITR2_NEXT:.*]], %[[ARR_BODY2_CTD:.*]] ]
// OGCG: %[[ITR2:.*]] = getelementptr inbounds [4 x %struct.NonTrivial], ptr %[[ITR2_BEGIN]], i64 %[[IDX2]]
// OGCG: %[[RHS_ITR2:.*]] = getelementptr inbounds nuw [3 x [4 x %struct.NonTrivial]], ptr %[[RHS_ITR1]], i64 0, i64 %[[IDX2]]
// OGCG: %[[ITR3_BEGIN:.*]] = getelementptr inbounds [4 x %struct.NonTrivial], ptr %[[ITR2]], i64 0, i64 0
// OGCG: br label %[[ARR_BODY3:.*]]

// OGCG: [[ARR_BODY3]]:
// OGCG: %[[IDX3:.*]] = phi i64 [ 0, %[[ARR_BODY2]] ], [ %[[ITR3_NEXT:.*]], %[[ARR_BODY3]] ]
// OGCG: %[[ITR3:.*]] = getelementptr inbounds %struct.NonTrivial, ptr %[[ITR3_BEGIN]], i64 %[[IDX3]]
// OGCG: %[[RHS_ITR3:.*]] = getelementptr inbounds nuw [4 x %struct.NonTrivial], ptr %[[RHS_ITR2]], i64 0, i64 %[[IDX3]]
// OGCG: call void @_ZN10NonTrivialC1ERKS_(ptr {{.*}}%[[ITR3]], ptr {{.*}}%[[RHS_ITR3]])
// OGCG: %[[ITR3_NEXT]] = add nuw i64 %[[IDX3]], 1
// OGCG: %[[COND3:.*]] = icmp eq i64 %[[ITR3_NEXT]], 4
// OGCG: br i1 %[[COND3]], label %[[ARR_BODY2_CTD]], label %[[ARR_BODY3]]

// OGCG: [[ARR_BODY2_CTD]]:
// OGCG: %[[ITR2_NEXT]] = add nuw i64 %[[IDX2]], 1
// OGCG: %[[COND2:.*]] = icmp eq i64 %[[ITR2_NEXT]], 3
// OGCG: br i1 %[[COND2]], label %[[ARR_BODY1_CTD]], label %[[ARR_BODY2]]

// OGCG: [[ARR_BODY1_CTD]]:
// OGCG: %[[ITR1_NEXT]] = add nuw i64 %[[IDX1]], 1
// OGCG: %[[COND1:.*]] = icmp eq i64 %[[ITR1_NEXT]], 2
// OGCG: br i1 %[[COND1]], label %[[END_BLOCK:.*]], label %[[ARR_BODY1]]

// OGCG: [[END_BLOCK]]:
// OGCG: ret void
extern "C" HasMultiDimArray make_multi_copy(const HasMultiDimArray &src) {
    return src;
}
