// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -mconstructor-aliases -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -mconstructor-aliases -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -mconstructor-aliases -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

struct A {
  ~A();
};

void test_temporary_dtor() {
  A();
}

// CIR: cir.func {{.*}} @_Z19test_temporary_dtorv()
// CIR:   %[[ALLOCA:.*]] = cir.alloca "agg.tmp.ensured" {{.*}} : !cir.ptr<!rec_A>
// CIR:   cir.call @_ZN1AD1Ev(%[[ALLOCA]]) nothrow : (!cir.ptr<!rec_A> {{.*}}) -> ()

// LLVM: define dso_local void @_Z19test_temporary_dtorv(){{.*}}
// LLVM:   %[[ALLOCA:.*]] = alloca %struct.A, align 1
// LLVM:   call void @_ZN1AD1Ev(ptr {{.*}} %[[ALLOCA]])

// OGCG: define dso_local void @_Z19test_temporary_dtorv()
// OGCG:   %[[ALLOCA:.*]] = alloca %struct.A, align 1
// OGCG:   call void @_ZN1AD1Ev(ptr {{.*}} %[[ALLOCA]])

struct B {
  int n;
  B(int n) : n(n) {}
  ~B() {}
};

bool make_temp(const B &) { return false; }
bool test_temp_or() { return make_temp(1) || make_temp(2); }

// The B(2) temporary lives in the right-hand side of the ||, which is only
// evaluated when the left-hand side is false. Its destructor is therefore a
// conditional cleanup guarded by an active flag, deferred to the enclosing
// full-expression scope so that the temporary lives to the end of the full
// expression rather than to the end of the right-hand operand.
//
// FIXME: The destruction order is wrong. B(1) is constructed first and B(2)
// second, so ~B(2) must run before ~B(1), as the OGCG checks below show. CIR
// runs them the other way around because the unconditional cleanup for B(1)
// gets its own nested cir.cleanup.scope that fires when the inner body ends,
// while the conditional cleanup for B(2) is deferred to the outer scope and
// fires later.
// CIR: cir.func{{.*}} @_Z12test_temp_orv()
// CIR:   %[[RET_ADDR:.*]] = cir.alloca "__retval" {{.*}} : !cir.ptr<!cir.bool>
// CIR:   %[[REF_TMP0:.*]] = cir.alloca "ref.tmp0" {{.*}} : !cir.ptr<!rec_B>
// CIR:   %[[REF_TMP1:.*]] = cir.alloca "ref.tmp1" {{.*}} : !cir.ptr<!rec_B>
// CIR:   %[[CLEANUP_COND:.*]] = cir.alloca "cleanup.cond" {{.*}} : !cir.ptr<!cir.bool>
// CIR:   cir.cleanup.scope {
// CIR:     %[[ONE:.*]] = cir.const #cir.int<1>
// CIR:     cir.call @_ZN1BC2Ei(%[[REF_TMP0]], %[[ONE]])
// CIR:     cir.cleanup.scope {
// CIR:       %[[MAKE_TEMP0:.*]] = cir.call @_Z9make_tempRK1B(%[[REF_TMP0]])
// CIR:       %[[FALSE:.*]] = cir.const #false
// CIR:       cir.store %[[FALSE]], %[[CLEANUP_COND]]
// CIR:       %[[TERNARY:.*]] = cir.ternary(%[[MAKE_TEMP0]], true {
// CIR:         %[[TRUE:.*]] = cir.const #true
// CIR:         cir.yield %[[TRUE]] : !cir.bool
// CIR:       }, false {
// CIR:         %[[TWO:.*]] = cir.const #cir.int<2>
// CIR:         cir.call @_ZN1BC2Ei(%[[REF_TMP1]], %[[TWO]])
// CIR:         %[[SET_TRUE:.*]] = cir.const #true
// CIR:         cir.store %[[SET_TRUE]], %[[CLEANUP_COND]]
// CIR:         %[[MAKE_TEMP1:.*]] = cir.call @_Z9make_tempRK1B(%[[REF_TMP1]])
// CIR:         cir.yield %[[MAKE_TEMP1]] : !cir.bool
// CIR:       })
// CIR:       cir.store{{.*}} %[[TERNARY]], %[[RET_ADDR]]
// CIR:       cir.yield
// FIXME: ~B(1) should run after the guarded ~B(2) below, not before it.
// CIR:     } cleanup normal {
// CIR:       cir.call @_ZN1BD2Ev(%[[REF_TMP0]])
// CIR:       cir.yield
// CIR:     }
// CIR:     cir.yield
// CIR:   } cleanup normal {
// CIR:     %[[IS_ACTIVE:.*]] = cir.load{{.*}} %[[CLEANUP_COND]]
// CIR:     cir.if %[[IS_ACTIVE]] {
// CIR:       cir.call @_ZN1BD2Ev(%[[REF_TMP1]])
// CIR:     }
// CIR:     cir.yield
// CIR:   }
// CIR:   %[[RETVAL:.*]] = cir.load{{.*}} %[[RET_ADDR]]
// CIR:   cir.return %[[RETVAL]]

// LLVM: define{{.*}} i1 @_Z12test_temp_orv(){{.*}} {
// LLVM:   %[[RETVAL:.*]] = alloca i8
// LLVM:   %[[REF_TMP0:.*]] = alloca %struct.B
// LLVM:   %[[REF_TMP1:.*]] = alloca %struct.B
// LLVM:   %[[CLEANUP_COND:.*]] = alloca i8
// LLVM:   call void @_ZN1BC2Ei(ptr {{.*}} %[[REF_TMP0]], i32 {{.*}} 1)
// LLVM:   %[[MAKE_TEMP0:.*]] = call {{.*}} i1 @_Z9make_tempRK1B(ptr {{.*}} %[[REF_TMP0]])
// LLVM:   store i8 0, ptr %[[CLEANUP_COND]]
// LLVM:   br i1 %[[MAKE_TEMP0]], label %[[TERN_TRUE:.*]], label %[[TERN_FALSE:.*]]
// LLVM: [[TERN_TRUE]]:
// LLVM:   br label %[[RESULT_BLOCK:.*]]
// LLVM: [[TERN_FALSE]]:
// LLVM:   call void @_ZN1BC2Ei(ptr {{.*}} %[[REF_TMP1]], i32 {{.*}} 2)
// LLVM:   store i8 1, ptr %[[CLEANUP_COND]]
// LLVM:   %[[MAKE_TEMP1:.*]] = call {{.*}} i1 @_Z9make_tempRK1B(ptr {{.*}} %[[REF_TMP1]])
// LLVM:   br label %[[RESULT_BLOCK]]
// LLVM: [[RESULT_BLOCK]]:
// LLVM:   %[[RESULT:.*]] = phi i1 [ %[[MAKE_TEMP1]], %[[TERN_FALSE]] ], [ true, %[[TERN_TRUE]] ]
// FIXME: ~B(1) should run after the guarded ~B(2) below. Compare the OGCG
// sequence, which destroys them in reverse order of construction.
// LLVM:   call void @_ZN1BD2Ev(ptr {{.*}} %[[REF_TMP0]])
// LLVM:   %[[FLAG_BYTE:.*]] = load i8, ptr %[[CLEANUP_COND]]
// LLVM:   %[[FLAG:.*]] = trunc i8 %[[FLAG_BYTE]] to i1
// LLVM:   br i1 %[[FLAG]], label %[[DTOR1:.*]], label %[[DONE:.*]]
// LLVM: [[DTOR1]]:
// LLVM:   call void @_ZN1BD2Ev(ptr {{.*}} %[[REF_TMP1]])
// LLVM:   br label %[[DONE]]

// OGCG: define {{.*}} i1 @_Z12test_temp_orv()
// OGCG: [[ENTRY:.*]]:
// OGCG:   %[[RETVAL:.*]] = alloca i1
// OGCG:   %[[REF_TMP0:.*]] = alloca %struct.B
// OGCG:   %[[REF_TMP1:.*]] = alloca %struct.B
// OGCG:   %[[CLEANUP_COND:.*]] = alloca i1
// OGCG:   call void @_ZN1BC2Ei(ptr {{.*}} %[[REF_TMP0]], i32 {{.*}} 1)
// OGCG:   %[[MAKE_TEMP0:.*]] = call {{.*}} i1 @_Z9make_tempRK1B(ptr {{.*}} %[[REF_TMP0]])
// OGCG:   store i1 false, ptr %cleanup.cond
// OGCG:   br i1 %[[MAKE_TEMP0]], label %[[LOR_END:.*]], label %[[LOR_RHS:.*]]
// OGCG: [[LOR_RHS]]:
// OGCG:   call void @_ZN1BC2Ei(ptr {{.*}} %[[REF_TMP1]], i32 {{.*}} 2)
// OGCG:   store i1 true, ptr %[[CLEANUP_COND]]
// OGCG:   %[[MAKE_TEMP1:.*]] = call {{.*}} i1 @_Z9make_tempRK1B(ptr {{.*}} %[[REF_TMP1]])
// OGCG:   br label %[[LOR_END]]
// OGCG: [[LOR_END]]:
// OGCG:    %[[PHI:.*]] = phi i1 [ true, %[[ENTRY]] ], [ %[[MAKE_TEMP1]], %[[LOR_RHS]] ]
// OGCG:   store i1 %[[PHI]], ptr %[[RETVAL]]
// OGCG:   %[[CLEANUP_IS_ACTIVE:.*]] = load i1, ptr %[[CLEANUP_COND]]
// OGCG:   br i1 %[[CLEANUP_IS_ACTIVE]], label %[[CLEANUP_ACTION:.*]], label %[[CLEANUP_DONE:.*]]
// OGCG: [[CLEANUP_ACTION]]:
// OGCG:   call void @_ZN1BD2Ev(ptr {{.*}} %[[REF_TMP1]])
// OGCG:   br label %[[CLEANUP_DONE]]
// OGCG: [[CLEANUP_DONE]]:
// OGCG:   call void @_ZN1BD2Ev(ptr {{.*}} %[[REF_TMP0]])

bool test_temp_and() { return make_temp(1) && make_temp(2); }

// As with test_temp_or, the B(2) temporary is created in the conditionally
// evaluated right-hand side, so its destructor is guarded by an active flag
// and deferred to the enclosing full-expression scope.
//
// FIXME: ~B(2) should run before ~B(1); see the OGCG checks below.
// CIR: cir.func{{.*}} @_Z13test_temp_andv()
// CIR:   %[[RET_ADDR:.*]] = cir.alloca "__retval" {{.*}} : !cir.ptr<!cir.bool>
// CIR:   %[[REF_TMP0:.*]] = cir.alloca "ref.tmp0" {{.*}} : !cir.ptr<!rec_B>
// CIR:   %[[REF_TMP1:.*]] = cir.alloca "ref.tmp1" {{.*}} : !cir.ptr<!rec_B>
// CIR:   %[[CLEANUP_COND:.*]] = cir.alloca "cleanup.cond" {{.*}} : !cir.ptr<!cir.bool>
// CIR:   cir.cleanup.scope {
// CIR:     %[[ONE:.*]] = cir.const #cir.int<1>
// CIR:     cir.call @_ZN1BC2Ei(%[[REF_TMP0]], %[[ONE]])
// CIR:     cir.cleanup.scope {
// CIR:       %[[MAKE_TEMP0:.*]] = cir.call @_Z9make_tempRK1B(%[[REF_TMP0]])
// CIR:       %[[FALSE:.*]] = cir.const #false
// CIR:       cir.store %[[FALSE]], %[[CLEANUP_COND]]
// CIR:       %[[TERNARY:.*]] = cir.ternary(%[[MAKE_TEMP0]], true {
// CIR:         %[[TWO:.*]] = cir.const #cir.int<2>
// CIR:         cir.call @_ZN1BC2Ei(%[[REF_TMP1]], %[[TWO]])
// CIR:         %[[SET_TRUE:.*]] = cir.const #true
// CIR:         cir.store %[[SET_TRUE]], %[[CLEANUP_COND]]
// CIR:         %[[MAKE_TEMP1:.*]] = cir.call @_Z9make_tempRK1B(%[[REF_TMP1]])
// CIR:         cir.yield %[[MAKE_TEMP1]] : !cir.bool
// CIR:       }, false {
// CIR:         %[[RES_FALSE:.*]] = cir.const #false
// CIR:         cir.yield %[[RES_FALSE]] : !cir.bool
// CIR:       })
// CIR:       cir.store{{.*}} %[[TERNARY]], %[[RET_ADDR]]
// CIR:       cir.yield
// CIR:     } cleanup normal {
// CIR:       cir.call @_ZN1BD2Ev(%[[REF_TMP0]])
// CIR:       cir.yield
// CIR:     }
// CIR:     cir.yield
// CIR:   } cleanup normal {
// CIR:     %[[IS_ACTIVE:.*]] = cir.load{{.*}} %[[CLEANUP_COND]]
// CIR:     cir.if %[[IS_ACTIVE]] {
// CIR:       cir.call @_ZN1BD2Ev(%[[REF_TMP1]])
// CIR:     }
// CIR:     cir.yield
// CIR:   }
// CIR:   %[[RETVAL:.*]] = cir.load{{.*}} %[[RET_ADDR]]
// CIR:   cir.return %[[RETVAL]]

// LLVM: define{{.*}} i1 @_Z13test_temp_andv(){{.*}} {
// LLVM:   %[[RETVAL:.*]] = alloca i8{{.*}}
// LLVM:   %[[REF_TMP0:.*]] = alloca %struct.B{{.*}}
// LLVM:   %[[REF_TMP1:.*]] = alloca %struct.B{{.*}}
// LLVM:   %[[CLEANUP_COND:.*]] = alloca i8{{.*}}
// LLVM:   call void @_ZN1BC2Ei(ptr {{.*}} %[[REF_TMP0]], i32 {{.*}} 1)
// LLVM:   %[[MAKE_TEMP0:.*]] = call {{.*}} i1 @_Z9make_tempRK1B(ptr {{.*}} %[[REF_TMP0]])
// LLVM:   store i8 0, ptr %[[CLEANUP_COND]]
// LLVM:   br i1 %[[MAKE_TEMP0]], label %[[TERN_TRUE:.*]], label %[[TERN_FALSE:.*]]
// LLVM: [[TERN_TRUE]]:
// LLVM:   call void @_ZN1BC2Ei(ptr {{.*}} %[[REF_TMP1]], i32 {{.*}} 2)
// LLVM:   store i8 1, ptr %[[CLEANUP_COND]]
// LLVM:   %[[MAKE_TEMP1:.*]] = call {{.*}} i1 @_Z9make_tempRK1B(ptr {{.*}} %[[REF_TMP1]])
// LLVM:   br label %[[RESULT_BLOCK:.*]]
// LLVM: [[TERN_FALSE]]:
// LLVM:   br label %[[RESULT_BLOCK]]
// LLVM: [[RESULT_BLOCK]]:
// LLVM:   %[[RESULT:.*]] = phi i1 [ false, %[[TERN_FALSE]] ], [ %[[MAKE_TEMP1]], %[[TERN_TRUE]] ]
// FIXME: ~B(1) should run after the guarded ~B(2) below; see OGCG.
// LLVM:   call void @_ZN1BD2Ev(ptr {{.*}} %[[REF_TMP0]])
// LLVM:   %[[FLAG_BYTE:.*]] = load i8, ptr %[[CLEANUP_COND]]
// LLVM:   %[[FLAG:.*]] = trunc i8 %[[FLAG_BYTE]] to i1
// LLVM:   br i1 %[[FLAG]], label %[[DTOR1:.*]], label %[[DONE:.*]]
// LLVM: [[DTOR1]]:
// LLVM:   call void @_ZN1BD2Ev(ptr {{.*}} %[[REF_TMP1]])
// LLVM:   br label %[[DONE]]
// LLVM: [[DONE]]:
// LLVM:   %[[RET_LOAD:.*]] = load i8, ptr %[[RETVAL]], align 1
// LLVM:   %[[RET_TRUNC:.*]] = trunc i8 %[[RET_LOAD]] to i1
// LLVM:   ret i1 %[[RET_TRUNC]]

// OGCG: define {{.*}} i1 @_Z13test_temp_andv()
// OGCG: [[ENTRY:.*]]:
// OGCG:   %[[RETVAL:.*]] = alloca i1
// OGCG:   %[[REF_TMP0:.*]] = alloca %struct.B
// OGCG:   %[[REF_TMP1:.*]] = alloca %struct.B
// OGCG:   %[[CLEANUP_COND:.*]] = alloca i1
// OGCG:   call void @_ZN1BC2Ei(ptr {{.*}} %[[REF_TMP0]], i32 {{.*}} 1)
// OGCG:   %[[MAKE_TEMP0:.*]] = call {{.*}} i1 @_Z9make_tempRK1B(ptr {{.*}} %[[REF_TMP0]])
// OGCG:   store i1 false, ptr %cleanup.cond
// OGCG:   br i1 %[[MAKE_TEMP0]], label %[[LAND_RHS:.*]], label %[[LAND_END:.*]]
// OGCG: [[LAND_RHS]]:
// OGCG:   call void @_ZN1BC2Ei(ptr {{.*}} %[[REF_TMP1]], i32 {{.*}} 2)
// OGCG:   store i1 true, ptr %[[CLEANUP_COND]]
// OGCG:   %[[MAKE_TEMP1:.*]] = call {{.*}} i1 @_Z9make_tempRK1B(ptr {{.*}} %[[REF_TMP1]])
// OGCG:   br label %[[LAND_END]]
// OGCG: [[LAND_END]]:
// OGCG:   %[[PHI:.*]] = phi i1 [ false, %[[ENTRY]] ], [ %[[MAKE_TEMP1]], %[[LAND_RHS]] ]
// OGCG:   store i1 %[[PHI]], ptr %[[RETVAL]]
// OGCG:   %[[CLEANUP_IS_ACTIVE:.*]] = load i1, ptr %[[CLEANUP_COND]]
// OGCG:   br i1 %[[CLEANUP_IS_ACTIVE]], label %[[CLEANUP_ACTION:.*]], label %[[CLEANUP_DONE:.*]]
// OGCG: [[CLEANUP_ACTION]]:
// OGCG:   call void @_ZN1BD2Ev(ptr {{.*}} %[[REF_TMP1]])
// OGCG:   br label %[[CLEANUP_DONE]]
// OGCG: [[CLEANUP_DONE]]:
// OGCG:   call void @_ZN1BD2Ev(ptr {{.*}} %[[REF_TMP0]])

struct C {
  ~C();
};

struct D {
  int n;
  C c;
  ~D() {}
};

void test_nested_dtor() {
  D d;
}

// CIR: cir.func{{.*}} @_Z16test_nested_dtorv()
// CIR:   cir.call @_ZN1DD2Ev(%{{.*}})

// LLVM: define {{.*}} void @_Z16test_nested_dtorv(){{.*}}
// LLVM:   call void @_ZN1DD2Ev(ptr {{.*}} %{{.*}})

// OGCG: define {{.*}} void @_Z16test_nested_dtorv()
// OGCG:   call void @_ZN1DD2Ev(ptr {{.*}} %{{.*}})

// CIR: cir.func {{.*}} @_ZN1DD2Ev
// CIR:   %[[BASE:.*]] = cir.cast bitcast %{{.*}} : !cir.ptr<!rec_D> -> !cir.ptr<!u8i>
// CIR:   %[[OFFSET:.*]] = cir.const #cir.int<4> : !u64i
// CIR:   %[[PTR:.*]] = cir.ptr_stride %[[BASE]], %[[OFFSET]] : (!cir.ptr<!u8i>, !u64i) -> !cir.ptr<!u8i>
// CIR:   %[[C:.*]] = cir.cast bitcast %[[PTR]] : !cir.ptr<!u8i> -> !cir.ptr<!rec_C>
// CIR:   cir.call @_ZN1CD1Ev(%[[C]])

// LLVM: define {{.*}} void @_ZN1DD2Ev
// LLVM:   %[[C:.*]] = getelementptr i8, ptr %{{.*}}, i64 4
// LLVM:   call void @_ZN1CD1Ev(ptr {{.*}} %[[C]])

// OGCG: define {{.*}} void @_ZN1DD2Ev
// OGCG:   %[[C:.*]] = getelementptr inbounds i8, ptr %{{.*}}, i64 4
// OGCG:   call void @_ZN1CD1Ev(ptr {{.*}} %[[C]])

struct E {
  ~E();
};

struct F : public E {
  int n;
  ~F() {}
};

void test_base_dtor_call() {
  F f;
}

// CIR: cir.func {{.*}} @_Z19test_base_dtor_callv()
//   cir.call @_ZN1FD2Ev(%{{.*}}) nothrow : (!cir.ptr<!rec_F> {{.*}}) -> ()

// LLVM: define {{.*}} void @_Z19test_base_dtor_callv(){{.*}}
// LLVM:   call void @_ZN1FD2Ev(ptr {{.*}} %{{.*}})

// OGCG: define {{.*}} void @_Z19test_base_dtor_callv()
// OGCG:   call void @_ZN1FD2Ev(ptr {{.*}} %{{.*}})

// CIR: cir.func {{.*}} @_ZN1FD2Ev
// CIR:   %[[BASE_E:.*]] = cir.base_class_addr %{{.*}} : !cir.ptr<!rec_F> nonnull [0] -> !cir.ptr<!rec_E>
// CIR:   cir.call @_ZN1ED2Ev(%[[BASE_E]]) nothrow : (!cir.ptr<!rec_E> {{.*}}) -> ()

// Because E is at offset 0 in F, there is no getelementptr needed.

// LLVM: define {{.*}} void @_ZN1FD2Ev
// LLVM:   call void @_ZN1ED2Ev(ptr {{.*}} %{{.*}})

// OGCG: define {{.*}} void @_ZN1FD2Ev
// OGCG:   call void @_ZN1ED2Ev(ptr {{.*}} %{{.*}})

struct G {
  G(int);
  ~G();
  G copy() const;
  bool operator==(const G &) const;
};

// Test the valesToReload handling in ScalarExprEmitter::VisitExprWithCleanups.
int test_temp_in_condition(G &obj) {
  if (obj.copy() == 1)
    return 1;
  return 0;
}

// CIR: cir.func {{.*}} @_Z22test_temp_in_conditionR1G(%[[ARG0:.*]]: !cir.ptr<!rec_G> {{.*}}) -> (!s32i {{.*}}) {{.*}} {
// CIR:   %[[OBJ:.*]] = cir.alloca "obj" {{.*}} init const : !cir.ptr<!cir.ptr<!rec_G>>
// CIR:   %[[RET_ADDR:.*]] = cir.alloca "__retval" {{.*}} : !cir.ptr<!s32i>
// CIR:   cir.store %[[ARG0]], %[[OBJ]]
// CIR:   cir.scope {
// CIR:     %[[REF_TMP0:.*]] = cir.alloca "ref.tmp0" {{.*}} : !cir.ptr<!rec_G>
// CIR:     %[[REF_TMP1:.*]] = cir.alloca "ref.tmp1" {{.*}} : !cir.ptr<!rec_G>
// CIR:     %[[CLEANUP_TMP:.*]] = cir.alloca "tmp.exprcleanup" {{.*}} : !cir.ptr<!cir.bool>
// CIR:     %[[LOAD_OBJ:.*]] = cir.load{{.*}} %[[OBJ]] : !cir.ptr<!cir.ptr<!rec_G>>, !cir.ptr<!rec_G>
// CIR:     cir.call @_ZNK1G4copyEv(%[[REF_TMP0]], %[[LOAD_OBJ]]) : (!cir.ptr<!rec_G> {{.*}}llvm.sret = !rec_G{{.*}}, !cir.ptr<!rec_G> {{.*}}) -> ()
// CIR:     cir.cleanup.scope {
// CIR:       %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR:       cir.call @_ZN1GC1Ei(%[[REF_TMP1]], %[[ONE]]) : (!cir.ptr<!rec_G> {{.*}}, !s32i {{.*}}) -> ()
// CIR:       cir.cleanup.scope {
// CIR:         %[[EQUAL:.*]] = cir.call @_ZNK1GeqERKS_(%[[REF_TMP0]], %[[REF_TMP1]]) : (!cir.ptr<!rec_G> {{.*}}, !cir.ptr<!rec_G> {{.*}}) -> (!cir.bool {{.*}})
// CIR:         cir.store{{.*}} %[[EQUAL]], %[[CLEANUP_TMP]]
// CIR:         cir.yield
// CIR:       } cleanup normal {
// CIR:         cir.call @_ZN1GD1Ev(%[[REF_TMP1]]) nothrow : (!cir.ptr<!rec_G> {{.*}}) -> ()
// CIR:         cir.yield
// CIR:       }
// CIR:       cir.yield
// CIR:     } cleanup normal {
// CIR:       cir.call @_ZN1GD1Ev(%[[REF_TMP0]]) nothrow : (!cir.ptr<!rec_G> {{.*}}) -> ()
// CIR:       cir.yield
// CIR:     }
// CIR:     %[[CONDITION:.*]] = cir.load{{.*}} %[[CLEANUP_TMP]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR:     cir.if %[[CONDITION]] {
// CIR:       %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR:       cir.store{{.*}} %[[ONE]], %[[RET_ADDR]]
// CIR:       %[[RETVAL:.*]] = cir.load{{.*}} %[[RET_ADDR]]
// CIR:       cir.return %[[RETVAL]]
// CIR:     }
// CIR:   }
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   cir.store{{.*}} %[[ZERO]], %[[RET_ADDR]]
// CIR:   %[[RETVAL:.*]] = cir.load{{.*}} %[[RET_ADDR]]
// CIR:   cir.return %[[RETVAL]]

// LLVM: define {{.*}} i32 @_Z22test_temp_in_conditionR1G(ptr {{.*}} %[[ARG0:.*]])
// LLVM:   %[[REF_TMP0:.*]] = alloca %struct.G
// LLVM:   %[[REF_TMP1:.*]] = alloca %struct.G
// LLVM:   %[[TMP_RESULT:.*]] = alloca i8
// LLVM:   %[[OBJ:.*]] = alloca ptr
// LLVM:   %[[RET_ADDR:.*]] = alloca i32
// LLVM:   store ptr %[[ARG0]], ptr %[[OBJ]]
// LLVM:   br label %[[SCOPE_BEGIN:.*]]
// LLVM: [[SCOPE_BEGIN]]:
// LLVM:   %[[LOAD_OBJ:.*]] = load ptr, ptr %[[OBJ]]
// LLVM:   call void @_ZNK1G4copyEv(ptr {{.*}} sret(%struct.G) {{.*}} %[[REF_TMP0]], ptr {{.*}} %[[LOAD_OBJ]])
// LLVM:   br label %[[CLEAN_SCOPE_ONE:.*]]
// LLVM: [[CLEAN_SCOPE_ONE]]:
// LLVM:   call void @_ZN1GC1Ei(ptr {{.*}} %[[REF_TMP1]], i32 {{.*}} 1)
// LLVM:   br label %[[CLEAN_SCOPE_TWO:.*]]
// LLVM: [[CLEAN_SCOPE_TWO]]:
// LLVM:   %[[EQUAL:.*]] = call noundef zeroext i1 @_ZNK1GeqERKS_(ptr {{.*}} %[[REF_TMP0]], ptr {{.*}} %[[REF_TMP1]])
// LLVM:   %[[ZEXT:.*]] = zext i1 %[[EQUAL]] to i8
// LLVM:   store i8 %[[ZEXT]], ptr %[[TMP_RESULT]]
// LLVM:   br label %[[CLEAN_SCOPE_TWO_CLEANUP:.*]]
// LLVM: [[CLEAN_SCOPE_TWO_CLEANUP]]:
// LLVM:   call void @_ZN1GD1Ev(ptr {{.*}} %[[REF_TMP1]])
// LLVM:   br label %[[EXIT_CLEAN_SCOPE_TWO:.*]]
// LLVM: [[EXIT_CLEAN_SCOPE_TWO]]:
// LLVM:   br label %[[CLEAN_SCOPE_ONE_CONTINUE:.*]]
// LLVM: [[CLEAN_SCOPE_ONE_CONTINUE]]:
// LLVM:   br label %[[CLEAN_SCOPE_ONE_CLEANUP:.*]]
// LLVM: [[CLEAN_SCOPE_ONE_CLEANUP]]:
// LLVM:   call void @_ZN1GD1Ev(ptr {{.*}} %[[REF_TMP0]])
// LLVM:   br label %[[EXIT_CLEAN_SCOPE_ONE:.*]]
// LLVM: [[EXIT_CLEAN_SCOPE_ONE]]:
// LLVM:   br label %[[SCOPE_CONTINUE:.*]]
// LLVM: [[SCOPE_CONTINUE]]:
// LLVM:   %[[LOAD_RESULT:.*]] = load i8, ptr %[[TMP_RESULT]]
// LLVM:   %[[TRUNC:.*]] = trunc i8 %[[LOAD_RESULT]] to i1
// LLVM:   br i1 %[[TRUNC]], label %[[TRUE_BLOCK:.*]], label %[[FALSE_BLOCK:.*]]
// LLVM: [[TRUE_BLOCK]]:
// LLVM:   store i32 1, ptr %[[RET_ADDR]]
// LLVM:   %[[RETVAL:.*]] = load i32, ptr %[[RET_ADDR]]
// LLVM:   ret i32 %[[RETVAL]]
// LLVM: [[FALSE_BLOCK]]:
// LLVM:   br label %[[EXIT_SCOPE:.*]]
// LLVM: [[EXIT_SCOPE]]:
// LLVM:   store i32 0, ptr %[[RET_ADDR]]
// LLVM:   %[[RETVAL:.*]] = load i32, ptr %[[RET_ADDR]]
// LLVM:   ret i32 %[[RETVAL]]

// OGCG: define {{.*}} i32 @_Z22test_temp_in_conditionR1G(ptr {{.*}} %[[ARG0:.*]])
// OGCG:   %[[RET_ADDR:.*]] = alloca i32
// OGCG:   %[[OBJ:.*]] = alloca ptr
// OGCG:   %[[REF_TMP0:.*]] = alloca %struct.G
// OGCG:   %[[REF_TMP1:.*]] = alloca %struct.G
// OGCG:   store ptr %[[ARG0]], ptr %[[OBJ]]
// OGCG:   %[[LOAD_OBJ:.*]] = load ptr, ptr %[[OBJ]]
// OGCG:   call void @_ZNK1G4copyEv(ptr {{.*}} %[[LOAD_OBJ]])
// OGCG:   call void @_ZN1GC1Ei(ptr {{.*}} %[[REF_TMP1]], i32 {{.*}} 1)
// OGCG:   %[[CALL:.*]] = call noundef zeroext i1 @_ZNK1GeqERKS_(ptr {{.*}} %[[REF_TMP0]], ptr {{.*}} %[[REF_TMP1]])
// OGCG:   call void @_ZN1GD1Ev(ptr {{.*}} %[[REF_TMP1]])
// OGCG:   call void @_ZN1GD1Ev(ptr {{.*}} %[[REF_TMP0]])
// OGCG:   br i1 %[[CALL]], label %[[IF_THEN:.*]], label %[[IF_END:.*]]
// OGCG: [[IF_THEN]]:
// OGCG:   store i32 1, ptr %[[RETVAL]]
// OGCG:   br label %[[RETURN:.*]]
// OGCG: [[IF_END]]:
// OGCG:   store i32 0, ptr %[[RETVAL]]
// OGCG:   br label %[[RETURN:.*]]
// OGCG: [[RETURN]]:
// OGCG:   %[[RETVAL:.*]] = load i32, ptr %[[RET_ADDR]]
// OGCG:   ret i32 %[[RETVAL]]

struct VirtualBase {
  ~VirtualBase();
};

struct Derived : virtual VirtualBase {
  ~Derived() {}
};

void test_base_dtor_call_virtual_base() {
  Derived d;
}

// Derived D1 (complete) destructor -- does call VirtualBase destructor

// CIR: cir.func {{.*}} @_ZN7DerivedD1Ev
// CIR:   %[[THIS:.*]] = cir.load %{{.*}}
// CIR:   %[[VTT:.*]] = cir.vtt.address_point @_ZTT7Derived, offset = 0 -> !cir.ptr<!cir.ptr<!void>>
// CIR:   cir.call @_ZN7DerivedD2Ev(%[[THIS]], %[[VTT]])
// CIR:   %[[VIRTUAL_BASE:.*]] = cir.base_class_addr %[[THIS]] : !cir.ptr<!rec_Derived> nonnull [0] -> !cir.ptr<!rec_VirtualBase>
// CIR:   cir.call @_ZN11VirtualBaseD2Ev(%[[VIRTUAL_BASE]])

// LLVM: define {{.*}} void @_ZN7DerivedD1Ev
// LLVM:   call void @_ZN7DerivedD2Ev(ptr {{.*}} %{{.*}}, ptr {{.*}} @_ZTT7Derived)
// LLVM:   call void @_ZN11VirtualBaseD2Ev(ptr {{.*}} %{{.*}})

// OGCG: define {{.*}} void @_ZN7DerivedD1Ev
// OGCG:   call void @_ZN7DerivedD2Ev(ptr {{.*}} %{{.*}}, ptr {{.*}} @_ZTT7Derived)
// OGCG:   call void @_ZN11VirtualBaseD2Ev(ptr {{.*}} %{{.*}})

// Derived D2 (base) destructor -- does not call VirtualBase destructor

// CIR:     cir.func {{.*}} @_ZN7DerivedD2Ev
// CIR-NOT:   cir.call{{.*}} @_ZN11VirtualBaseD2Ev
// CIR:       cir.return

// LLVM:     define {{.*}} void @_ZN7DerivedD2Ev
// LLVM-NOT:   call{{.*}} @_ZN11VirtualBaseD2Ev
// LLVM:       ret

// OGCG:     define {{.*}} void @_ZN7DerivedD2Ev
// OGCG-NOT:   call{{.*}} @_ZN11VirtualBaseD2Ev
// OGCG:       ret
