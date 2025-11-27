// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o -  2>&1 | FileCheck --check-prefixes=CIR-BEFORE-LPP %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

struct S {
    S();
};

void foo() {
    S s[42];
}

// CIR-BEFORE-LPP: cir.func dso_local @_Z3foov()
// CIR-BEFORE-LPP:   %[[ARRAY:.*]] = cir.alloca !cir.array<!rec_S x 42>, !cir.ptr<!cir.array<!rec_S x 42>>, ["s", init]
// CIR-BEFORE-LPP:   cir.array.ctor %[[ARRAY]] : !cir.ptr<!cir.array<!rec_S x 42>> {
// CIR-BEFORE-LPP:    ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_S>):
// CIR-BEFORE-LPP:      cir.call @_ZN1SC1Ev(%[[ARG]]) : (!cir.ptr<!rec_S>) -> ()
// CIR-BEFORE-LPP:      cir.yield
// CIR-BEFORE-LPP:    }
// CIR-BEFORE-LPP:   cir.return
// CIR-BEFORE-LPP: }

// CIR: cir.func dso_local @_Z3foov()
// CIR:   %[[ARRAY:.*]] = cir.alloca !cir.array<!rec_S x 42>, !cir.ptr<!cir.array<!rec_S x 42>>, ["s", init]
// CIR:   %[[CONST42:.*]] = cir.const #cir.int<42> : !u64i
// CIR:   %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ARRAY]] : !cir.ptr<!cir.array<!rec_S x 42>> -> !cir.ptr<!rec_S>
// CIR:   %[[END_PTR:.*]] = cir.ptr_stride %[[DECAY]], %[[CONST42]] : (!cir.ptr<!rec_S>, !u64i) -> !cir.ptr<!rec_S>
// CIR:   %[[ITER:.*]] = cir.alloca !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>, ["__array_idx"]
// CIR:   cir.store %[[DECAY]], %[[ITER]] : !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>
// CIR:   cir.do {
// CIR:     %[[CURRENT:.*]] = cir.load %[[ITER]] : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
// CIR:     cir.call @_ZN1SC1Ev(%[[CURRENT]]) : (!cir.ptr<!rec_S>) -> ()
// CIR:     %[[CONST1:.*]] = cir.const #cir.int<1> : !u64i
// CIR:     %[[NEXT:.*]] = cir.ptr_stride %[[CURRENT]], %[[CONST1]] : (!cir.ptr<!rec_S>, !u64i) -> !cir.ptr<!rec_S>
// CIR:     cir.store %[[NEXT]], %[[ITER]] : !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>
// CIR:     cir.yield
// CIR:   } while {
// CIR:     %[[CURRENT2:.*]] = cir.load %[[ITER]] : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
// CIR:     %[[CMP:.*]] = cir.cmp(ne, %[[CURRENT2]], %[[END_PTR]]) : !cir.ptr<!rec_S>, !cir.bool
// CIR:     cir.condition(%[[CMP]])
// CIR:   }
// CIR:   cir.return
// CIR: }

// LLVM: define dso_local void @_Z3foov()
// LLVM: %[[ARRAY:.*]] = alloca [42 x %struct.S]
// LLVM: %[[START:.*]] = getelementptr %struct.S, ptr %[[ARRAY]], i32 0
// LLVM: %[[END:.*]] = getelementptr %struct.S, ptr %[[START]], i64 42
// LLVM: %[[ITER:.*]] = alloca ptr
// LLVM: store ptr %[[START]], ptr %[[ITER]]
// LLVM: br label %[[LOOP:.*]]
// LLVM: [[COND:.*]]:
// LLVM: %[[CURRENT_CHECK:.*]] = load ptr, ptr %[[ITER]]
// LLVM: %[[DONE:.*]] = icmp ne ptr %[[CURRENT_CHECK]], %[[END]]
// LLVM: br i1 %[[DONE]], label %[[LOOP]], label %[[EXIT:.*]]
// LLVM: [[LOOP]]:
// LLVM: %[[CURRENT:.*]] = load ptr, ptr %[[ITER]]
// LLVM: call void @_ZN1SC1Ev(ptr %[[CURRENT]])
// LLVM: %[[NEXT:.*]] = getelementptr %struct.S, ptr %[[CURRENT]], i64 1
// LLVM: store ptr %[[NEXT]], ptr %[[ITER]]
// LLVM: br label %[[COND]]
// LLVM: [[EXIT]]:
// LLVM: ret void

// OGCG: define dso_local void @_Z3foov()
// OGCG: %[[ARRAY:.*]] = alloca [42 x %struct.S]
// OGCG: %[[START:.*]] = getelementptr{{.*}} %struct.S{{.*}}
// OGCG: %[[END:.*]] = getelementptr{{.*}} %struct.S{{.*}} i64 42
// OGCG: br label %[[LOOP:.*]]
// OGCG: [[LOOP]]:
// OGCG: %[[CURRENT:.*]] = phi ptr [ %[[START]], %{{.*}} ], [ %[[NEXT:.*]], %[[LOOP]] ]
// OGCG: call void @_ZN1SC1Ev(ptr{{.*}})
// OGCG: %[[NEXT]] = getelementptr{{.*}} %struct.S{{.*}} i64 1
// OGCG: %[[DONE:.*]] = icmp eq ptr %[[NEXT]], %[[END]]
// OGCG: br i1 %[[DONE]], label %[[EXIT:.*]], label %[[LOOP]]
// OGCG: [[EXIT]]:
// OGCG: ret void

void zero_sized() {
    S s[0];
}

// CIR-BEFORE-LPP:     cir.func dso_local @_Z10zero_sizedv()
// CIR-BEFORE-LPP:       cir.alloca !cir.array<!rec_S x 0>, !cir.ptr<!cir.array<!rec_S x 0>>, ["s"]
// CIR-BEFORE-LPP-NOT:   cir.array.ctor
// CIR-BEFORE-LPP:       cir.return

// CIR:     cir.func dso_local @_Z10zero_sizedv()
// CIR:       cir.alloca !cir.array<!rec_S x 0>, !cir.ptr<!cir.array<!rec_S x 0>>, ["s"]
// CIR-NOT:   cir.do
// CIR-NOT:   cir.call @_ZN1SC1Ev
// CIR:       cir.return

// LLVM:     define dso_local void @_Z10zero_sizedv()
// LLVM:       alloca [0 x %struct.S]
// LLVM-NOT:   call void @_ZN1SC1Ev
// LLVM:       ret void

// OGCG:     define dso_local void @_Z10zero_sizedv()
// OGCG:       alloca [0 x %struct.S]
// OGCG-NOT:   call void @_ZN1SC1Ev
// OGCG:       ret void

void multi_dimensional() {
    S s[3][5];
}

// CIR-BEFORE-LPP:     cir.func{{.*}} @_Z17multi_dimensionalv()
// CIR-BEFORE-LPP:       %[[S:.*]] = cir.alloca !cir.array<!cir.array<!rec_S x 5> x 3>, !cir.ptr<!cir.array<!cir.array<!rec_S x 5> x 3>>, ["s", init]
// CIR-BEFORE-LPP:       %[[FLAT:.*]] = cir.cast bitcast %[[S]] : !cir.ptr<!cir.array<!cir.array<!rec_S x 5> x 3>> -> !cir.ptr<!cir.array<!rec_S x 15>>
// CIR-BEFORE-LPP:       cir.array.ctor %[[FLAT]] : !cir.ptr<!cir.array<!rec_S x 15>> {
// CIR-BEFORE-LPP:        ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_S>):
// CIR-BEFORE-LPP:          cir.call @_ZN1SC1Ev(%[[ARG]]) : (!cir.ptr<!rec_S>) -> ()
// CIR-BEFORE-LPP:          cir.yield
// CIR-BEFORE-LPP:       }
// CIR-BEFORE-LPP:       cir.return

// CIR:     cir.func{{.*}} @_Z17multi_dimensionalv()
// CIR:       %[[S:.*]] = cir.alloca !cir.array<!cir.array<!rec_S x 5> x 3>, !cir.ptr<!cir.array<!cir.array<!rec_S x 5> x 3>>, ["s", init]
// CIR:       %[[CONST15:.*]] = cir.const #cir.int<15> : !u64i
// CIR:       %[[DECAY:.*]] = cir.cast array_to_ptrdecay {{.*}} : !cir.ptr<!cir.array<!rec_S x 15>> -> !cir.ptr<!rec_S>
// CIR:       %[[END_PTR:.*]] = cir.ptr_stride %[[DECAY]], %[[CONST15]] : (!cir.ptr<!rec_S>, !u64i) -> !cir.ptr<!rec_S>
// CIR:       %[[ITER:.*]] = cir.alloca !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>, ["__array_idx"]
// CIR:       cir.store %[[DECAY]], %[[ITER]] : !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>
// CIR:       cir.do {
// CIR:         %[[CURRENT:.*]] = cir.load %[[ITER]] : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
// CIR:         cir.call @_ZN1SC1Ev(%[[CURRENT]]) : (!cir.ptr<!rec_S>) -> ()
// CIR:         %[[CONST1:.*]] = cir.const #cir.int<1> : !u64i
// CIR:         %[[NEXT:.*]] = cir.ptr_stride %[[CURRENT]], %[[CONST1]] : (!cir.ptr<!rec_S>, !u64i) -> !cir.ptr<!rec_S>
// CIR:         cir.store %[[NEXT]], %[[ITER]] : !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>
// CIR:         cir.yield
// CIR:       } while {
// CIR:         %[[CURRENT2:.*]] = cir.load %[[ITER]] : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
// CIR:         %[[CMP:.*]] = cir.cmp(ne, %[[CURRENT2]], %[[END_PTR]]) : !cir.ptr<!rec_S>, !cir.bool
// CIR:         cir.condition(%[[CMP]])
// CIR:       }
// CIR:       cir.return

// LLVM:     define{{.*}} @_Z17multi_dimensionalv()
// LLVM:       %[[S:.*]] = alloca [3 x [5 x %struct.S]]
// LLVM:       %[[START:.*]] = getelementptr %struct.S, ptr %[[S]], i32 0
// LLVM:       %[[END:.*]] = getelementptr %struct.S, ptr %[[START]], i64 15
// LLVM:       %[[ITER:.*]] = alloca ptr
// LLVM:       store ptr %[[START]], ptr %[[ITER]]
// LLVM:       br label %[[LOOP:.*]]
// LLVM:     [[COND:.*]]:
// LLVM:       %[[CURRENT_CHECK:.*]] = load ptr, ptr %[[ITER]]
// LLVM:       %[[DONE:.*]] = icmp ne ptr %[[CURRENT_CHECK]], %[[END]]
// LLVM:       br i1 %[[DONE]], label %[[LOOP]], label %[[EXIT:.*]]
// LLVM:     [[LOOP]]:
// LLVM:       %[[CURRENT:.*]] = load ptr, ptr %[[ITER]]
// LLVM:       call void @_ZN1SC1Ev(ptr %[[CURRENT]])
// LLVM:       %[[NEXT:.*]] = getelementptr %struct.S, ptr %[[CURRENT]], i64 1
// LLVM:       store ptr %[[NEXT]], ptr %[[ITER]]
// LLVM:       br label %[[COND]]
// LLVM:     [[EXIT]]:
// LLVM:       ret void

// OGCG:     define{{.*}} @_Z17multi_dimensionalv()
// OGCG:       %[[S:.*]] = alloca [3 x [5 x %struct.S]]
// OGCG:       %[[START:.*]] = getelementptr{{.*}} %struct.S{{.*}}
// OGCG:       %[[END:.*]] = getelementptr{{.*}} %struct.S{{.*}} i64 15
// OGCG:       br label %[[LOOP:.*]]
// OGCG:     [[LOOP]]:
// OGCG:       %[[CURRENT:.*]] = phi ptr [ %[[START]], %{{.*}} ], [ %[[NEXT:.*]], %[[LOOP]] ]
// OGCG:       call void @_ZN1SC1Ev(ptr{{.*}})
// OGCG:       %[[NEXT]] = getelementptr{{.*}} %struct.S{{.*}} i64 1
// OGCG:       %[[DONE:.*]] = icmp eq ptr %[[NEXT]], %[[END]]
// OGCG:       br i1 %[[DONE]], label %[[EXIT:.*]], label %[[LOOP]]
// OGCG:     [[EXIT]]:
// OGCG:       ret void
