// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir  2> %t-before-lp.cir
// RUN: FileCheck --input-file=%t-before-lp.cir %s -check-prefix=CIR-BEFORE-LPP
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

struct S {
    ~S();
};

void test_cleanup_array() {
    S s[42];
}

// CIR-BEFORE-LPP: cir.func{{.*}} @_Z18test_cleanup_arrayv()
// CIR-BEFORE-LPP:   %[[S:.*]] = cir.alloca !cir.array<!rec_S x 42>, !cir.ptr<!cir.array<!rec_S x 42>>, ["s"]
// CIR-BEFORE-LPP:   cir.array.dtor %[[S]] : !cir.ptr<!cir.array<!rec_S x 42>> {
// CIR-BEFORE-LPP:   ^bb0(%arg0: !cir.ptr<!rec_S>
// CIR-BEFORE-LPP:     cir.call @_ZN1SD1Ev(%arg0) nothrow : (!cir.ptr<!rec_S>) -> ()
// CIR-BEFORE-LPP:     cir.yield
// CIR-BEFORE-LPP:   }
// CIR-BEFORE-LPP:   cir.return

// CIR: cir.func{{.*}} @_Z18test_cleanup_arrayv()
// CIR:   %[[S:.*]] = cir.alloca !cir.array<!rec_S x 42>, !cir.ptr<!cir.array<!rec_S x 42>>, ["s"]
// CIR:   %[[CONST41:.*]] = cir.const #cir.int<41> : !u64i
// CIR:   %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[S]] : !cir.ptr<!cir.array<!rec_S x 42>> -> !cir.ptr<!rec_S>
// CIR:   %[[END_PTR:.*]] = cir.ptr_stride %[[DECAY]], %[[CONST41]] : (!cir.ptr<!rec_S>, !u64i) -> !cir.ptr<!rec_S>
// CIR:   %[[ITER:.*]] = cir.alloca !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>, ["__array_idx"]
// CIR:   cir.store %[[END_PTR]], %[[ITER]] : !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>
// CIR:   cir.do {
// CIR:     %[[CURRENT:.*]] = cir.load %[[ITER]] : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
// CIR:     cir.call @_ZN1SD1Ev(%[[CURRENT]]) nothrow : (!cir.ptr<!rec_S>) -> ()
// CIR:     %[[CONST_MINUS1:.*]] = cir.const #cir.int<-1> : !s64i
// CIR:     %[[NEXT:.*]] = cir.ptr_stride %[[CURRENT]], %[[CONST_MINUS1]] : (!cir.ptr<!rec_S>, !s64i) -> !cir.ptr<!rec_S>
// CIR:     cir.store %[[NEXT]], %[[ITER]] : !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>
// CIR:     cir.yield
// CIR:   } while {
// CIR:     %[[CURRENT2:.*]] = cir.load %[[ITER]] : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
// CIR:     %[[CMP:.*]] = cir.cmp(ne, %[[CURRENT2]], %[[DECAY]])
// CIR:     cir.condition(%[[CMP]])
// CIR:   }
// CIR:   cir.return

// LLVM: define{{.*}} void @_Z18test_cleanup_arrayv()
// LLVM:   %[[ARRAY:.*]] = alloca [42 x %struct.S]
// LLVM:   %[[START:.*]] = getelementptr %struct.S, ptr %[[ARRAY]], i32 0
// LLVM:   %[[END:.*]] = getelementptr %struct.S, ptr %[[START]], i64 41
// LLVM:   %[[ITER:.*]] = alloca ptr
// LLVM:   store ptr %[[END]], ptr %[[ITER]]
// LLVM:   br label %[[LOOP:.*]]
// LLVM: [[COND:.*]]:
// LLVM:   %[[CURRENT_CHECK:.*]] = load ptr, ptr %[[ITER]]
// LLVM:   %[[DONE:.*]] = icmp ne ptr %[[CURRENT_CHECK]], %[[START]]
// LLVM:   br i1 %[[DONE]], label %[[LOOP]], label %[[EXIT:.*]]
// LLVM: [[LOOP]]:
// LLVM:   %[[CURRENT:.*]] = load ptr, ptr %[[ITER]]
// LLVM:   call void @_ZN1SD1Ev(ptr %[[CURRENT]])
// LLVM:   %[[NEXT:.*]] = getelementptr %struct.S, ptr %[[CURRENT]], i64 -1
// LLVM:   store ptr %[[NEXT]], ptr %[[ITER]]
// LLVM:   br label %[[COND]]
// LLVM: [[EXIT]]:
// LLVM:   ret void

// OGCG: define{{.*}} void @_Z18test_cleanup_arrayv()
// OGCG:   %[[ARRAY:.*]] = alloca [42 x %struct.S]
// OGCG:   %[[START:.*]] = getelementptr{{.*}} %struct.S{{.*}}
// OGCG:   %[[END:.*]] = getelementptr{{.*}} %struct.S{{.*}} i64 42
// OGCG:   br label %[[LOOP:.*]]
// OGCG: [[LOOP]]:
// OGCG:   %[[NEXT:.*]] = phi ptr [ %[[END]], %{{.*}} ], [ %[[LAST:.*]], %[[LOOP]] ]
// OGCG:   %[[LAST]] = getelementptr{{.*}} %struct.S{{.*}}, ptr %[[NEXT]], i64 -1
// OGCG:   call void @_ZN1SD1Ev(ptr{{.*}} %[[LAST]])
// OGCG:   %[[DONE:.*]] = icmp eq ptr %[[LAST]], %[[START]]
// OGCG:   br i1 %[[DONE]], label %[[EXIT:.*]], label %[[LOOP]]
// OGCG: [[EXIT]]:
// OGCG:   ret void

void test_cleanup_zero_length_array() {
    S s[0];
}

// CIR-BEFORE-LPP:     cir.func{{.*}} @_Z30test_cleanup_zero_length_arrayv()
// CIR-BEFORE-LPP:       %[[S:.*]] = cir.alloca !cir.array<!rec_S x 0>, !cir.ptr<!cir.array<!rec_S x 0>>, ["s"]
// CIR-BEFORE-LPP-NOT:   cir.array.dtor
// CIR-BEFORE-LPP:       cir.return

// CIR:     cir.func{{.*}} @_Z30test_cleanup_zero_length_arrayv()
// CIR:       %[[S:.*]] = cir.alloca !cir.array<!rec_S x 0>, !cir.ptr<!cir.array<!rec_S x 0>>, ["s"]
// CIR-NOT:   cir.do
// CIR-NOT:   cir.call @_ZN1SD1Ev
// CIR:       cir.return

// LLVM:     define{{.*}} void @_Z30test_cleanup_zero_length_arrayv()
// LLVM:       alloca [0 x %struct.S]
// LLVM-NOT:   call void @_ZN1SD1Ev
// LLVM:       ret void

// OGCG:     define{{.*}} void @_Z30test_cleanup_zero_length_arrayv()
// OGCG:       alloca [0 x %struct.S]
// OGCG-NOT:   call void @_ZN1SD1Ev
// OGCG:       ret void

void multi_dimensional() {
    S s[3][5];
}

// CIR-BEFORE-LPP:     cir.func{{.*}} @_Z17multi_dimensionalv()
// CIR-BEFORE-LPP:       %[[S:.*]] = cir.alloca !cir.array<!cir.array<!rec_S x 5> x 3>, !cir.ptr<!cir.array<!cir.array<!rec_S x 5> x 3>>, ["s"]
// CIR-BEFORE-LPP:       %[[FLAT:.*]] = cir.cast bitcast %[[S]] : !cir.ptr<!cir.array<!cir.array<!rec_S x 5> x 3>> -> !cir.ptr<!cir.array<!rec_S x 15>>
// CIR-BEFORE-LPP:       cir.array.dtor %[[FLAT]] : !cir.ptr<!cir.array<!rec_S x 15>> {
// CIR-BEFORE-LPP:       ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_S>):
// CIR-BEFORE-LPP:         cir.call @_ZN1SD1Ev(%[[ARG]]) nothrow : (!cir.ptr<!rec_S>) -> ()
// CIR-BEFORE-LPP:         cir.yield
// CIR-BEFORE-LPP:       }
// CIR-BEFORE-LPP:       cir.return

// CIR:     cir.func{{.*}} @_Z17multi_dimensionalv()
// CIR:       %[[S:.*]] = cir.alloca !cir.array<!cir.array<!rec_S x 5> x 3>, !cir.ptr<!cir.array<!cir.array<!rec_S x 5> x 3>>, ["s"]
// CIR:       %[[FLAT:.*]] = cir.cast bitcast %[[S]] : !cir.ptr<!cir.array<!cir.array<!rec_S x 5> x 3>> -> !cir.ptr<!cir.array<!rec_S x 15>>
// CIR:       %[[CONST14:.*]] = cir.const #cir.int<14> : !u64i
// CIR:       %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[FLAT]] : !cir.ptr<!cir.array<!rec_S x 15>> -> !cir.ptr<!rec_S>
// CIR:       %[[END_PTR:.*]] = cir.ptr_stride %[[DECAY]], %[[CONST14]] : (!cir.ptr<!rec_S>, !u64i) -> !cir.ptr<!rec_S>
// CIR:       %[[ITER:.*]] = cir.alloca !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>, ["__array_idx"]
// CIR:       cir.store %[[END_PTR]], %[[ITER]] : !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>
// CIR:       cir.do {
// CIR:         %[[CUR:.*]] = cir.load %[[ITER]] : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
// CIR:         cir.call @_ZN1SD1Ev(%[[CUR]]) nothrow : (!cir.ptr<!rec_S>) -> ()
// CIR:         %[[NEG1:.*]] = cir.const #cir.int<-1> : !s64i
// CIR:         %[[PREV:.*]] = cir.ptr_stride %[[CUR]], %[[NEG1]] : (!cir.ptr<!rec_S>, !s64i) -> !cir.ptr<!rec_S>
// CIR:         cir.store %[[PREV]], %[[ITER]] : !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>
// CIR:         cir.yield
// CIR:       } while {
// CIR:         %[[CHK:.*]] = cir.load %[[ITER]] : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
// CIR:         %[[CMP:.*]] = cir.cmp(ne, %[[CHK]], %[[DECAY]])
// CIR:         cir.condition(%[[CMP]])
// CIR:       }
// CIR:       cir.return

// LLVM:     define{{.*}} void @_Z17multi_dimensionalv()
// LLVM:       %[[S:.*]] = alloca [3 x [5 x %struct.S]]
// LLVM:       %[[START:.*]] = getelementptr %struct.S, ptr %[[S]], i32 0
// LLVM:       %[[END:.*]] = getelementptr %struct.S, ptr %[[START]], i64 14
// LLVM:       %[[ITER:.*]] = alloca ptr
// LLVM:       store ptr %[[END]], ptr %[[ITER]]
// LLVM:       br label %[[LOOP:.*]]
// LLVM: [[COND:.*]]:
// LLVM:       %[[CURRENT_CHECK:.*]] = load ptr, ptr %[[ITER]]
// LLVM:       %[[DONE:.*]] = icmp ne ptr %[[CURRENT_CHECK]], %[[START]]
// LLVM:       br i1 %[[DONE]], label %[[LOOP]], label %[[EXIT:.*]]
// LLVM: [[LOOP]]:
// LLVM:       %[[CUR:.*]] = load ptr, ptr %[[ITER]]
// LLVM:       call void @_ZN1SD1Ev(ptr %[[CUR]])
// LLVM:       %[[PREV:.*]] = getelementptr %struct.S, ptr %[[CUR]], i64 -1
// LLVM:       store ptr %[[PREV]], ptr %[[ITER]]
// LLVM:       br label %[[COND]]
// LLVM: [[EXIT]]:
// LLVM:       ret void

// OGCG:     define{{.*}} void @_Z17multi_dimensionalv()
// OGCG:       %[[ARRAY:.*]] = alloca [3 x [5 x %struct.S]]
// OGCG:       %[[START:.*]] = getelementptr{{.*}} %struct.S{{.*}}
// OGCG:       %[[END:.*]] = getelementptr{{.*}} %struct.S{{.*}} i64 15
// OGCG:       br label %[[LOOP:.*]]
// OGCG: [[LOOP]]:
// OGCG:       %[[NEXT:.*]] = phi ptr [ %[[END]], %{{.*}} ], [ %[[LAST:.*]], %[[LOOP]] ]
// OGCG:       %[[LAST]] = getelementptr{{.*}} %struct.S{{.*}}, ptr %[[NEXT]], i64 -1
// OGCG:       call void @_ZN1SD1Ev(ptr{{.*}} %[[LAST]])
// OGCG:       %[[DONE:.*]] = icmp eq ptr %[[LAST]], %[[START]]
// OGCG:       br i1 %[[DONE]], label %[[EXIT:.*]], label %[[LOOP]]
// OGCG: [[EXIT]]:
// OGCG:       ret void

