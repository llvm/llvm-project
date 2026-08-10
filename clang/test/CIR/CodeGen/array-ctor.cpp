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

// CIR-BEFORE-LPP: cir.func {{.*}} @_Z3foov()
// CIR-BEFORE-LPP:   %[[ARRAY:.*]] = cir.alloca "s" {{.*}} init : !cir.ptr<!cir.array<!rec_S x 42>>
// CIR-BEFORE-LPP:   cir.array.ctor %[[ARRAY]] : !cir.ptr<!cir.array<!rec_S x 42>> {
// CIR-BEFORE-LPP:    ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_S>):
// CIR-BEFORE-LPP:      cir.call @_ZN1SC1Ev(%[[ARG]]) : (!cir.ptr<!rec_S>{{.*}}) -> ()
// CIR-BEFORE-LPP:    }
// CIR-BEFORE-LPP:   cir.return
// CIR-BEFORE-LPP: }

// CIR: cir.func {{.*}} @_Z3foov()
// CIR:   %[[ARRAY:.*]] = cir.alloca "s" {{.*}} init : !cir.ptr<!cir.array<!rec_S x 42>>
// CIR:   %[[CONST42:.*]] = cir.const #cir.int<42> : !u64i
// CIR:   %[[DECAY:.*]] = cir.cast array_to_ptrdecay %[[ARRAY]] : !cir.ptr<!cir.array<!rec_S x 42>> -> !cir.ptr<!rec_S>
// CIR:   %[[END_PTR:.*]] = cir.ptr_stride %[[DECAY]], %[[CONST42]] : (!cir.ptr<!rec_S>, !u64i) -> !cir.ptr<!rec_S>
// CIR:   %[[ITER:.*]] = cir.alloca "__array_idx" {{.*}} : !cir.ptr<!cir.ptr<!rec_S>>
// CIR:   cir.store %[[DECAY]], %[[ITER]] : !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>
// CIR:   cir.do {
// CIR:     %[[CURRENT:.*]] = cir.load %[[ITER]] : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
// CIR:     cir.call @_ZN1SC1Ev(%[[CURRENT]]) : (!cir.ptr<!rec_S>{{.*}}) -> ()
// CIR:     %[[CONST1:.*]] = cir.const #cir.int<1> : !u64i
// CIR:     %[[NEXT:.*]] = cir.ptr_stride %[[CURRENT]], %[[CONST1]] : (!cir.ptr<!rec_S>, !u64i) -> !cir.ptr<!rec_S>
// CIR:     cir.store %[[NEXT]], %[[ITER]] : !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>
// CIR:     cir.yield
// CIR:   } while {
// CIR:     %[[CURRENT2:.*]] = cir.load %[[ITER]] : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
// CIR:     %[[CMP:.*]] = cir.cmp ne %[[CURRENT2]], %[[END_PTR]] : !cir.ptr<!rec_S>
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
// LLVM: call void @_ZN1SC1Ev(ptr{{.*}} %[[CURRENT]])
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

// CIR-BEFORE-LPP:     cir.func {{.*}} @_Z10zero_sizedv()
// CIR-BEFORE-LPP:       cir.alloca "s" {{.*}} : !cir.ptr<!cir.array<!rec_S x 0>>
// CIR-BEFORE-LPP-NOT:   cir.array.ctor
// CIR-BEFORE-LPP:       cir.return

// CIR:     cir.func {{.*}} @_Z10zero_sizedv()
// CIR:       cir.alloca "s" {{.*}} : !cir.ptr<!cir.array<!rec_S x 0>>
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
// CIR-BEFORE-LPP:       %[[S:.*]] = cir.alloca "s" {{.*}} init : !cir.ptr<!cir.array<!cir.array<!rec_S x 5> x 3>>
// CIR-BEFORE-LPP:       %[[FLAT:.*]] = cir.cast bitcast %[[S]] : !cir.ptr<!cir.array<!cir.array<!rec_S x 5> x 3>> -> !cir.ptr<!cir.array<!rec_S x 15>>
// CIR-BEFORE-LPP:       cir.array.ctor %[[FLAT]] : !cir.ptr<!cir.array<!rec_S x 15>> {
// CIR-BEFORE-LPP:        ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_S>):
// CIR-BEFORE-LPP:          cir.call @_ZN1SC1Ev(%[[ARG]]) : (!cir.ptr<!rec_S>{{.*}}) -> ()
// CIR-BEFORE-LPP:       }
// CIR-BEFORE-LPP:       cir.return

// CIR:     cir.func{{.*}} @_Z17multi_dimensionalv()
// CIR:       %[[S:.*]] = cir.alloca "s" {{.*}} init : !cir.ptr<!cir.array<!cir.array<!rec_S x 5> x 3>>
// CIR:       %[[CONST15:.*]] = cir.const #cir.int<15> : !u64i
// CIR:       %[[DECAY:.*]] = cir.cast array_to_ptrdecay {{.*}} : !cir.ptr<!cir.array<!rec_S x 15>> -> !cir.ptr<!rec_S>
// CIR:       %[[END_PTR:.*]] = cir.ptr_stride %[[DECAY]], %[[CONST15]] : (!cir.ptr<!rec_S>, !u64i) -> !cir.ptr<!rec_S>
// CIR:       %[[ITER:.*]] = cir.alloca "__array_idx" {{.*}} : !cir.ptr<!cir.ptr<!rec_S>>
// CIR:       cir.store %[[DECAY]], %[[ITER]] : !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>
// CIR:       cir.do {
// CIR:         %[[CURRENT:.*]] = cir.load %[[ITER]] : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
// CIR:         cir.call @_ZN1SC1Ev(%[[CURRENT]]) : (!cir.ptr<!rec_S>{{.*}}) -> ()
// CIR:         %[[CONST1:.*]] = cir.const #cir.int<1> : !u64i
// CIR:         %[[NEXT:.*]] = cir.ptr_stride %[[CURRENT]], %[[CONST1]] : (!cir.ptr<!rec_S>, !u64i) -> !cir.ptr<!rec_S>
// CIR:         cir.store %[[NEXT]], %[[ITER]] : !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>
// CIR:         cir.yield
// CIR:       } while {
// CIR:         %[[CURRENT2:.*]] = cir.load %[[ITER]] : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
// CIR:         %[[CMP:.*]] = cir.cmp ne %[[CURRENT2]], %[[END_PTR]] : !cir.ptr<!rec_S>
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
// LLVM:       call void @_ZN1SC1Ev(ptr{{.*}} %[[CURRENT]])
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

struct Temp {
  ~Temp();
};

struct CausesTemp {
  CausesTemp(Temp = Temp());
};

void TempInArray() {
  CausesTemp ct[42];
}

// CIR-BEFORE-LPP: cir.func {{.*}} @_Z11TempInArrayv()
// CIR-BEFORE-LPP:   %[[ARRAY:.*]] = cir.alloca "ct" {{.*}} init : !cir.ptr<!cir.array<!rec_CausesTemp x 42>>
// CIR-BEFORE-LPP:   %[[TMP:.*]] = cir.alloca "agg.tmp0" {{.*}} : !cir.ptr<!rec_Temp>
// CIR-BEFORE-LPP:   cir.array.ctor %[[ARRAY]] : !cir.ptr<!cir.array<!rec_CausesTemp x 42>> {
// CIR-BEFORE-LPP:    ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_CausesTemp>):
// CIR-BEFORE-LPP:      cir.cleanup.scope {
// CIR-BEFORE-LPP:        %[[LOAD:.*]] = cir.load {{.*}} %[[TMP]] : !cir.ptr<!rec_Temp>, !rec_Temp
// CIR-BEFORE-LPP:        cir.call @_ZN10CausesTempC1E4Temp(%[[ARG]], %[[LOAD]])
// CIR-BEFORE-LPP:        cir.yield
// CIR-BEFORE-LPP:      } cleanup normal {
// CIR-BEFORE-LPP:        cir.call @_ZN4TempD1Ev(%[[TMP]]) nothrow
// CIR-BEFORE-LPP:        cir.yield
// CIR-BEFORE-LPP:      }
// CIR-BEFORE-LPP:    }
// CIR-BEFORE-LPP:   cir.return
// CIR-BEFORE-LPP: }

// CIR-LABEL: cir.func {{.*}} @_Z11TempInArrayv()
// CIR:        %[[TMP:.*]] = cir.alloca "agg.tmp0" {{.*}} : !cir.ptr<!rec_Temp>
// CIR:        cir.do {
// CIR-NEXT:     %[[CURRENT:.*]] = cir.load %[[ITER:.*]] : !cir.ptr<!cir.ptr<!rec_CausesTemp>>, !cir.ptr<!rec_CausesTemp>
// CIR-NEXT:     cir.cleanup.scope {
// CIR-NEXT:       %[[LOAD:.*]] = cir.load {{.*}} %[[TMP]] : !cir.ptr<!rec_Temp>, !rec_Temp
// CIR-NEXT:       cir.call @_ZN10CausesTempC1E4Temp(%[[CURRENT]], %[[LOAD]])
// CIR-NEXT:       cir.yield
// CIR-NEXT:     } cleanup normal {
// CIR-NEXT:       cir.call @_ZN4TempD1Ev(%[[TMP]]) nothrow
// CIR-NEXT:       cir.yield
// CIR-NEXT:     }
// CIR-NEXT:     %[[CONST1:.*]] = cir.const #cir.int<1> : !u64i
// CIR-NEXT:     %[[NEXT:.*]] = cir.ptr_stride %[[CURRENT]], %[[CONST1]] : (!cir.ptr<!rec_CausesTemp>, !u64i) -> !cir.ptr<!rec_CausesTemp>
// CIR-NEXT:     cir.store %[[NEXT]], %[[ITER]] : !cir.ptr<!rec_CausesTemp>, !cir.ptr<!cir.ptr<!rec_CausesTemp>>
// CIR-NEXT:     cir.yield
// CIR-NEXT:   } while {
// CIR:        }

// LLVM-LABEL: define {{.*}}void @_Z11TempInArrayv()
// LLVM:       %[[TMP:.*]] = alloca %struct.Temp
// LLVM:       %[[ITER:.*]] = alloca ptr
// LLVM:       br label %[[DO_BR:.*]]
// LLVM:       [[DO_BR]]:
// LLVM:       %[[CURRENT:.*]] = load ptr, ptr %[[ITER]]
// LLVM:       br label %[[CONSTRUCT_BR:.*]]
// LLVM:       [[CONSTRUCT_BR]]:
// LLVM:       %[[LOAD:.*]] = load %struct.Temp, ptr %[[TMP]]
// LLVM:       call void @_ZN10CausesTempC1E4Temp(ptr {{.*}}%[[CURRENT]], %struct.Temp %[[LOAD]])
// LLVM:       br label %[[CLEANUP_BR:.*]]
// LLVM:       [[CLEANUP_BR]]:
// LLVM:       call void @_ZN4TempD1Ev({{.*}}[[TMP]])

// OGCG-LABEL: define {{.*}}void @_Z11TempInArrayv()
// OGCG:       %[[TMP:.*]] = alloca %struct.Temp
// OGCG:       br label %[[LOOP:.*]]
// OGCG:       [[LOOP]]:
// OGCG:       %[[CURRENT:.*]] = phi ptr
// OGCG:       call void @_ZN10CausesTempC1E4Temp(ptr {{.*}}%[[CURRENT]], ptr{{.*}}[[TMP]])
// OGCG:       call void @_ZN4TempD1Ev({{.*}}[[TMP]])

struct Temp2 {
  Temp2();
  ~Temp2();
};

struct CausesTemp2 {
  CausesTemp2(Temp2 = Temp2());
  ~CausesTemp2();
};

void Temp2InArray() {
  CausesTemp2 ct2[42];
}

// CIR-BEFORE-LPP-LABEL: cir.func {{.*}} @_Z12Temp2InArrayv()
// CIR-BEFORE-LPP-NEXT:   %[[ARRAY:.*]] = cir.alloca "ct2" {{.*}} init : !cir.ptr<!cir.array<!rec_CausesTemp2 x 42>>
// CIR-BEFORE-LPP-NEXT:   %[[TMP:.*]] = cir.alloca "agg.tmp0" {{.*}} : !cir.ptr<!rec_Temp2>
// CIR-BEFORE-LPP-NEXT:   cir.array.ctor %[[ARRAY]] : !cir.ptr<!cir.array<!rec_CausesTemp2 x 42>> {
// CIR-BEFORE-LPP-NEXT:    ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_CausesTemp2>):
// CIR-BEFORE-LPP-NEXT:      cir.call @_ZN5Temp2C1Ev(%[[TMP]])
// CIR-BEFORE-LPP-NEXT:      cir.cleanup.scope {
// CIR-BEFORE-LPP-NEXT:        %[[LOAD:.*]] = cir.load {{.*}} %[[TMP]] : !cir.ptr<!rec_Temp2>, !rec_Temp2
// CIR-BEFORE-LPP-NEXT:        cir.call @_ZN11CausesTemp2C1E5Temp2(%[[ARG]], %[[LOAD]])
// CIR-BEFORE-LPP-NEXT:        cir.yield
// CIR-BEFORE-LPP-NEXT:      } cleanup normal {
// CIR-BEFORE-LPP-NEXT:        cir.call @_ZN5Temp2D1Ev(%[[TMP]]) nothrow
// CIR-BEFORE-LPP-NEXT:        cir.yield
// CIR-BEFORE-LPP-NEXT:      }
// CIR-BEFORE-LPP-NEXT:    }
// CIR-BEFORE-LPP-NEXT:    cir.cleanup.scope {
// CIR-BEFORE-LPP-NEXT:      cir.yield
// CIR-BEFORE-LPP-NEXT:    } cleanup normal {
// CIR-BEFORE-LPP-NEXT:      cir.array.dtor %[[ARRAY]] : !cir.ptr<!cir.array<!rec_CausesTemp2 x 42>> {
// CIR-BEFORE-LPP-NEXT:      ^bb0(%[[ARG:.*]]: !cir.ptr<!rec_CausesTemp2>):
// CIR-BEFORE-LPP-NEXT:        cir.call @_ZN11CausesTemp2D1Ev(%[[ARG]]) nothrow : ({{.*}})
// CIR-BEFORE-LPP-NEXT:      }
// CIR-BEFORE-LPP-NEXT:      cir.yield
// CIR-BEFORE-LPP-NEXT:    }
// CIR-BEFORE-LPP-NEXT:   cir.return
// CIR-BEFORE-LPP-NEXT: }

// CIR-BEFORE-LPP: module @



// CIR-LABEL: cir.func {{.*}} @_Z12Temp2InArrayv()
// CIR:        %[[TMP:.*]] = cir.alloca "agg.tmp0" {{.*}} : !cir.ptr<!rec_Temp2>
// CIR:        cir.do {
// CIR-NEXT:     %[[CURRENT:.*]] = cir.load %[[ITER:.*]] : !cir.ptr<!cir.ptr<!rec_CausesTemp2>>, !cir.ptr<!rec_CausesTemp2>
// CIR-NEXT:     cir.call @_ZN5Temp2C1Ev(%[[TMP]])
// CIR-NEXT:     cir.cleanup.scope {
// CIR-NEXT:       %[[LOAD:.*]] = cir.load {{.*}} %[[TMP]] : !cir.ptr<!rec_Temp2>, !rec_Temp2
// CIR-NEXT:         cir.call @_ZN11CausesTemp2C1E5Temp2(%[[CURRENT]], %[[LOAD]])
// CIR-NEXT:         cir.yield
// CIR-NEXT:       } cleanup normal {
// CIR-NEXT:         cir.call @_ZN5Temp2D1Ev(%[[TMP]]) nothrow
// CIR-NEXT:         cir.yield
// CIR-NEXT:     }
// CIR-NEXT:     %[[CONST1:.*]] = cir.const #cir.int<1> : !u64i
// CIR-NEXT:     %[[NEXT:.*]] = cir.ptr_stride %[[CURRENT]], %[[CONST1]] : (!cir.ptr<!rec_CausesTemp2>, !u64i) -> !cir.ptr<!rec_CausesTemp2>
// CIR-NEXT:     cir.store %[[NEXT]], %[[ITER]] : !cir.ptr<!rec_CausesTemp2>, !cir.ptr<!cir.ptr<!rec_CausesTemp2>>
// CIR-NEXT:     cir.yield
// CIR-NEXT:   } while {
// CIR:        }
// Array dtor:
// CIR-NEXT:   cir.cleanup.scope {
// CIR-NEXT:     cir.yield
// CIR-NEXT:   } cleanup normal {
// CIR:         %[[IDX:.*]] = cir.alloca "__array_idx" {{.*}} : !cir.ptr<!cir.ptr<!rec_CausesTemp2>>
// CIR:         cir.do {
// CIR-NEXT:       %[[LOAD_IDX:.*]] = cir.load %[[IDX]] : !cir.ptr<!cir.ptr<!rec_CausesTemp2>>, !cir.ptr<!rec_CausesTemp2>
// CIR-NEXT:       %[[NEG_1:.*]] = cir.const #cir.int<-1> : !s64i
// CIR-NEXT:       %[[STRIDE:.*]] = cir.ptr_stride %[[LOAD_IDX]], %[[NEG_1]]
// CIR:            cir.call @_ZN11CausesTemp2D1Ev(%[[STRIDE]]) nothrow 
// CIR-NEXT:       cir.yield
// CIR-NEXT:     } while {
// CIR:          }


// LLVM-LABEL: define {{.*}}void @_Z12Temp2InArrayv()
// LLVM:       %[[DTOR_ITR:.*]] = alloca ptr
// LLVM:       %[[TMP:.*]] = alloca %struct.Temp2
// LLVM:       %[[ITER:.*]] = alloca ptr
// LLVM:       br label %[[DO_BR:.*]]
// LLVM:       [[DO_BR]]:
// LLVM:       %[[CURRENT:.*]] = load ptr, ptr %[[ITER]]
// LLVM:       call void @_ZN5Temp2C1Ev({{.*}}%[[TMP]])
// LLVM:       br label %[[CONSTRUCT_BR:.*]]
// LLVM:       [[CONSTRUCT_BR]]:
// LLVM:       %[[LOAD:.*]] = load %struct.Temp2, ptr %[[TMP]]
// LLVM:       call void @_ZN11CausesTemp2C1E5Temp2(ptr {{.*}}%[[CURRENT]], %struct.Temp2 %[[LOAD]])
// LLVM:       br label %[[CLEANUP_BR:.*]]
// LLVM:       [[CLEANUP_BR]]:
// LLVM:       call void @_ZN5Temp2D1Ev({{.*}}[[TMP]])
// LLVM:       %[[DTOR_ELT:.*]] = getelementptr %struct.CausesTemp2, ptr %{{.*}}, i64 -1
// LLVM:       call void @_ZN11CausesTemp2D1Ev(ptr {{.*}}%[[DTOR_ELT]])

// OGCG-LABEL: define {{.*}}void @_Z12Temp2InArrayv()
// OGCG:       %[[TMP:.*]] = alloca %struct.Temp
// OGCG:       br label %[[LOOP:.*]]
// OGCG:       [[LOOP]]:
// OGCG:       %[[CURRENT:.*]] = phi ptr
// OGCG:       call void @_ZN5Temp2C1Ev(ptr {{.*}}%[[TMP]])
// OGCG-NEXT:  call void @_ZN11CausesTemp2C1E5Temp2(ptr {{.*}}%[[CURRENT]], ptr{{.*}}[[TMP]])
// OGCG-NEXT:  call void @_ZN5Temp2D1Ev({{.*}}[[TMP]])
// OGCG:       %[[DTOR_ELT:.*]] = getelementptr inbounds %struct.CausesTemp2, ptr %{{.*}}, i64 -1
// OGCG-NEXT:  call void @_ZN11CausesTemp2D1Ev(ptr {{.*}}%[[DTOR_ELT]])
