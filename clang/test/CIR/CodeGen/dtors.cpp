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
// CIR:   %[[ALLOCA:.*]] = cir.alloca !rec_A, !cir.ptr<!rec_A>, ["agg.tmp.ensured"]
// CIR:   cir.call @_ZN1AD1Ev(%[[ALLOCA]]) nothrow : (!cir.ptr<!rec_A> {{.*}}) -> ()

// LLVM: define dso_local void @_Z19test_temporary_dtorv(){{.*}}
// LLVM:   %[[ALLOCA:.*]] = alloca %struct.A, i64 1, align 1
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

// CIR: cir.func{{.*}} @_Z12test_temp_orv()
// CIR:   %[[RET_ADDR:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["__retval"]
// CIR:   cir.scope {
// CIR:     %[[REF_TMP0:.*]] = cir.alloca !rec_B, !cir.ptr<!rec_B>, ["ref.tmp0"]
// CIR:     %[[ONE:.*]] = cir.const #cir.int<1>
// CIR:     cir.call @_ZN1BC2Ei(%[[REF_TMP0]], %[[ONE]])
// CIR:     cir.cleanup.scope {
// CIR:       %[[MAKE_TEMP0:.*]] = cir.call @_Z9make_tempRK1B(%[[REF_TMP0]])
// CIR:       %[[TERNARY:.*]] = cir.ternary(%[[MAKE_TEMP0]], true {
// CIR:         %[[TRUE:.*]] = cir.const #true
// CIR:         cir.yield %[[TRUE]] : !cir.bool
// CIR:       }, false {
// CIR:         %[[REF_TMP1:.*]] = cir.alloca !rec_B, !cir.ptr<!rec_B>, ["ref.tmp1"]
// CIR:         %[[CLEANUP_TMP:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["tmp.exprcleanup"]
// CIR:         %[[TWO:.*]] = cir.const #cir.int<2>
// CIR:         cir.call @_ZN1BC2Ei(%[[REF_TMP1]], %[[TWO]])
// CIR:         cir.cleanup.scope {
// CIR:           %[[MAKE_TEMP1:.*]] = cir.call @_Z9make_tempRK1B(%[[REF_TMP1]]) : (!cir.ptr<!rec_B>
// CIR:           cir.store{{.*}} %[[MAKE_TEMP1]], %[[CLEANUP_TMP]]
// CIR:           cir.yield
// CIR:         } cleanup  normal {
// CIR:           cir.call @_ZN1BD2Ev(%[[REF_TMP1]])
// CIR:           cir.yield
// CIR:         }
// CIR:         %[[TERNARY_TMP:.*]] = cir.load{{.*}} %[[CLEANUP_TMP]]
// CIR:         cir.yield %[[TERNARY_TMP]] : !cir.bool
// CIR:       })
// CIR:       cir.store{{.*}} %[[TERNARY]], %[[RET_ADDR]]
// CIR:       cir.yield
// CIR:     } cleanup  normal {
// CIR:       cir.call @_ZN1BD2Ev(%[[REF_TMP0]])
// CIR:       cir.yield
// CIR:     }
// CIR:   }
// CIR:   %[[RETVAL:.*]] = cir.load{{.*}} %[[RET_ADDR]]
// CIR:   cir.return %[[RETVAL]]

// LLVM: define{{.*}} i1 @_Z12test_temp_orv(){{.*}} {
// LLVM:   %[[REF_TMP0:.*]] = alloca %struct.B
// LLVM:   %[[REF_TMP1:.*]] = alloca %struct.B
// LLVM:   %[[TMP_RESULT1:.*]] = alloca i8
// LLVM:   %[[TMP_RESULT2:.*]] = alloca i8
// LLVM:   br label %[[LOR_BEGIN:.*]]
// LLVM: [[LOR_BEGIN]]:
// LLVM:   call void @_ZN1BC2Ei(ptr {{.*}} %[[REF_TMP0]], i32 {{.*}} 1)
// LLVM:   br label %[[SCOPE_BEGIN:.*]]
// LLVM: [[SCOPE_BEGIN]]:
// LLVM:   %[[MAKE_TEMP0:.*]] = call {{.*}} i1 @_Z9make_tempRK1B(ptr {{.*}} %[[REF_TMP0]])
// LLVM:   br i1 %[[MAKE_TEMP0]], label %[[TERN_TRUE:.*]], label %[[TERN_FALSE:.*]]
// LLVM: [[TERN_TRUE]]:
// LLVM:   br label %[[RESULT_BLOCK:.*]]
// LLVM: [[TERN_FALSE]]:
// LLVM:   call void @_ZN1BC2Ei(ptr {{.*}} %[[REF_TMP1]], i32 {{.*}} 2)
// LLVM:   br label %[[FALSE_CLEANUP_SCOPE:.*]]
// LLVM: [[FALSE_CLEANUP_SCOPE]]:
// LLVM:   %[[MAKE_TEMP1:.*]] = call {{.*}} i1 @_Z9make_tempRK1B(ptr {{.*}} %[[REF_TMP1]])
// LLVM:   %[[ZEXT:.*]] = zext i1 %[[MAKE_TEMP1]] to i8
// LLVM:   store i8 %[[ZEXT]], ptr %[[TMP_RESULT1]]
// LLVM:   br label %[[FALSE_CLEANUP:.*]]
// LLVM: [[FALSE_CLEANUP]]:
// LLVM:   call void @_ZN1BD2Ev(ptr {{.*}} %[[REF_TMP1]])
// LLVM:   br label %[[FALSE_END:.*]]
// LLVM: [[FALSE_END]]:
// LLVM:   br label %[[FALSE_RESULT:.*]]
// LLVM: [[FALSE_RESULT]]:
// LLVM:   %[[LOAD_TEMP1:.*]] = load i8, ptr %[[TMP_RESULT1]]
// LLVM:   %[[TRUNC:.*]] = trunc i8 %[[LOAD_TEMP1]] to i1
// LLVM:   br label %[[RESULT_BLOCK:.*]]
// LLVM: [[RESULT_BLOCK]]:
// LLVM:   %[[RESULT:.*]] = phi i1 [ %[[TRUNC]], %[[FALSE_RESULT]] ], [ true, %[[TERN_TRUE]] ]
// LLVM:   br label %[[TERN_END:.*]]
// LLVM: [[TERN_END]]:
// LLVM:   %[[ZEXT_RESULT:.*]] = zext i1 %[[RESULT]] to i8
// LLVM:   store i8 %[[ZEXT_RESULT]], ptr %[[TMP_RESULT2]]
// LLVM:   br label %[[LOR_END:.*]]
// LLVM: [[LOR_END]]:
// LLVM:   call void @_ZN1BD2Ev(ptr {{.*}} %[[REF_TMP0]])

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

// CIR: cir.func{{.*}} @_Z13test_temp_andv()
// CIR:   %[[RET_ADDR:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["__retval"]
// CIR:   cir.scope {
// CIR:     %[[REF_TMP0:.*]] = cir.alloca !rec_B, !cir.ptr<!rec_B>, ["ref.tmp0"]
// CIR:     %[[ONE:.*]] = cir.const #cir.int<1>
// CIR:     cir.call @_ZN1BC2Ei(%[[REF_TMP0]], %[[ONE]])
// CIR:     cir.cleanup.scope {
// CIR:       %[[MAKE_TEMP0:.*]] = cir.call @_Z9make_tempRK1B(%[[REF_TMP0]])
// CIR:       %[[TERNARY:.*]] = cir.ternary(%[[MAKE_TEMP0]], true {
// CIR:         %[[REF_TMP1:.*]] = cir.alloca !rec_B, !cir.ptr<!rec_B>, ["ref.tmp1"]
// CIR:         %[[CLEANUP_TMP:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["tmp.exprcleanup"]
// CIR:         %[[TWO:.*]] = cir.const #cir.int<2>
// CIR:         cir.call @_ZN1BC2Ei(%[[REF_TMP1]], %[[TWO]])
// CIR:         cir.cleanup.scope {
// CIR:           %[[MAKE_TEMP1:.*]] = cir.call @_Z9make_tempRK1B(%[[REF_TMP1]]) : (!cir.ptr<!rec_B>
// CIR:           cir.store{{.*}} %[[MAKE_TEMP1]], %[[CLEANUP_TMP]]
// CIR:           cir.yield
// CIR:         } cleanup  normal {
// CIR:           cir.call @_ZN1BD2Ev(%[[REF_TMP1]])
// CIR:           cir.yield
// CIR:         }
// CIR:         %[[TERNARY_TMP:.*]] = cir.load{{.*}} %[[CLEANUP_TMP]]
// CIR:         cir.yield %[[TERNARY_TMP]] : !cir.bool
// CIR:       }, false {
// CIR:         %[[FALSE:.*]] = cir.const #false
// CIR:         cir.yield %[[FALSE]] : !cir.bool
// CIR:       })
// CIR:       cir.store{{.*}} %[[TERNARY]], %[[RET_ADDR]]
// CIR:       cir.yield
// CIR:     } cleanup  normal {
// CIR:       cir.call @_ZN1BD2Ev(%[[REF_TMP0]])
// CIR:       cir.yield
// CIR:     }
// CIR:   }
// CIR:   %[[RETVAL:.*]] = cir.load{{.*}} %[[RET_ADDR]]
// CIR:   cir.return %[[RETVAL]]

// LLVM: define{{.*}} i1 @_Z13test_temp_andv(){{.*}} {
// LLVM:   %[[REF_TMP0:.*]] = alloca %struct.B{{.*}}
// LLVM:   %[[REF_TMP1:.*]] = alloca %struct.B{{.*}}
// LLVM:   %[[TMP_RESULT:.*]] = alloca i8{{.*}}
// LLVM:   %[[TMP_RESULT2:.*]] = alloca i8{{.*}}
// LLVM:   br label %[[LAND_BEGIN:.*]]
// LLVM: [[LAND_BEGIN]]:
// LLVM:   call void @_ZN1BC2Ei(ptr {{.*}} %[[REF_TMP0]], i32 {{.*}} 1)
// LLVM:   br label %[[SCOPE_BEGIN:.*]]
// LLVM: [[SCOPE_BEGIN]]:
// LLVM:   %[[MAKE_TEMP0:.*]] = call {{.*}} i1 @_Z9make_tempRK1B(ptr {{.*}} %[[REF_TMP0]])
// LLVM:   br i1 %[[MAKE_TEMP0]], label %[[TERN_TRUE:.*]], label %[[TERN_FALSE:.*]]
// LLVM: [[TERN_TRUE]]:
// LLVM:   call void @_ZN1BC2Ei(ptr {{.*}} %[[REF_TMP1]], i32 {{.*}} 2)
// LLVM:   br label %[[TRUE_CLEANUP_SCOPE:.*]]
// LLVM: [[TRUE_CLEANUP_SCOPE]]:
// LLVM:   %[[MAKE_TEMP1:.*]] = call {{.*}} i1 @_Z9make_tempRK1B(ptr {{.*}} %[[REF_TMP1]])
// LLVM:   %[[ZEXT_MAKE_TEMP1:.*]] = zext i1 %[[MAKE_TEMP1]] to i8
// LLVM:   store i8 %[[ZEXT_MAKE_TEMP1]], ptr %[[TMP_RESULT]], align 1
// LLVM:   br label %[[TRUE_CLEANUP:.*]]
// LLVM: [[TRUE_CLEANUP]]:
// LLVM:   call void @_ZN1BD2Ev(ptr {{.*}} %[[REF_TMP1]])
// LLVM:   br label %[[TRUE_END:.*]]
// LLVM: [[TRUE_END]]:
// LLVM:   br label %[[TRUE_RESULT:.*]]
// LLVM: [[TRUE_RESULT]]:
// LLVM:   %[[TMP_LOAD:.*]] = load i8, ptr %[[TMP_RESULT]], align 1
// LLVM:   %[[TMP_TRUNC:.*]] = trunc i8 %[[TMP_LOAD]] to i1
// LLVM:   br label %[[RESULT_BLOCK:.*]]
// LLVM: [[TERN_FALSE]]:
// LLVM:   br label %[[RESULT_BLOCK]]
// LLVM: [[RESULT_BLOCK]]:
// LLVM:   %[[RESULT:.*]] = phi i1 [ false, %[[TERN_FALSE]] ], [ %[[TMP_TRUNC]], %[[TRUE_RESULT]] ]
// LLVM:   br label %[[TERN_END:.*]]
// LLVM: [[TERN_END]]:
// LLVM:   %[[ZEXT_RESULT:.*]] = zext i1 %[[RESULT]] to i8
// LLVM:   store i8 %[[ZEXT_RESULT]], ptr %[[TMP_RESULT2]], align 1
// LLVM:   br label %[[LAND_END:.*]]
// LLVM: [[LAND_END]]:
// LLVM:   call void @_ZN1BD2Ev(ptr {{.*}} %[[REF_TMP0]])
// LLVM:   br label %[[LAND_END2:.*]]
// LLVM: [[LAND_END2]]:
// LLVM:   br label %[[LAND_END3:.*]]
// LLVM: [[LAND_END3]]:
// LLVM:   br label %[[RETURN_BLOCK:.*]]
// LLVM: [[RETURN_BLOCK]]:
// LLVM:   %[[TMP2_LOAD:.*]] = load i8, ptr %[[TMP_RESULT2]], align 1
// LLVM:   %[[TMP2_TRUNC:.*]] = trunc i8 %[[TMP2_LOAD]] to i1
// LLVM:   ret i1 %[[TMP2_TRUNC]]

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

// CIR: cir.func {{.*}} @_ZN1DD2Ev
// CIR:   %[[C:.*]] = cir.get_member %{{.*}}[1] {name = "c"}
// CIR:   cir.call @_ZN1CD1Ev(%[[C]])

// LLVM: define {{.*}} void @_ZN1DD2Ev
// LLVM:   %[[C:.*]] = getelementptr %struct.D, ptr %{{.*}}, i32 0, i32 1
// LLVM:   call void @_ZN1CD1Ev(ptr {{.*}} %[[C]])

// This destructor is defined after the calling function in OGCG.

void test_nested_dtor() {
  D d;
}

// CIR: cir.func{{.*}} @_Z16test_nested_dtorv()
// CIR:   cir.call @_ZN1DD2Ev(%{{.*}})

// LLVM: define {{.*}} void @_Z16test_nested_dtorv(){{.*}}
// LLVM:   call void @_ZN1DD2Ev(ptr {{.*}} %{{.*}})

// OGCG: define {{.*}} void @_Z16test_nested_dtorv()
// OGCG:   call void @_ZN1DD2Ev(ptr {{.*}} %{{.*}})

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

// CIR: cir.func {{.*}} @_ZN1FD2Ev
// CIR:   %[[BASE_E:.*]] = cir.base_class_addr %{{.*}} : !cir.ptr<!rec_F> nonnull [0] -> !cir.ptr<!rec_E>
// CIR:   cir.call @_ZN1ED2Ev(%[[BASE_E]]) nothrow : (!cir.ptr<!rec_E> {{.*}}) -> ()

// Because E is at offset 0 in F, there is no getelementptr needed.

// LLVM: define {{.*}} void @_ZN1FD2Ev
// LLVM:   call void @_ZN1ED2Ev(ptr {{.*}} %{{.*}})

// This destructor is defined after the calling function in OGCG.

void test_base_dtor_call() {
  F f;
}

// CIR: cir.func {{.*}} @_Z19test_base_dtor_callv()
//   cir.call @_ZN1FD2Ev(%{{.*}}) nothrow : (!cir.ptr<!rec_F> {{.*}}) -> ()

// LLVM: define {{.*}} void @_Z19test_base_dtor_callv(){{.*}}
// LLVM:   call void @_ZN1FD2Ev(ptr {{.*}} %{{.*}})

// OGCG: define {{.*}} void @_Z19test_base_dtor_callv()
// OGCG:   call void @_ZN1FD2Ev(ptr {{.*}} %{{.*}})

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

// CIR: cir.func {{.*}} @_Z22test_temp_in_conditionR1G(%[[ARG0:.*]]: !cir.ptr<!rec_G> {{.*}}) -> (!s32i {{.*}}) {
// CIR:   %[[OBJ:.*]] = cir.alloca !cir.ptr<!rec_G>, !cir.ptr<!cir.ptr<!rec_G>>, ["obj", init, const]
// CIR:   %[[RET_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR:   cir.store %[[ARG0]], %[[OBJ]]
// CIR:   cir.scope {
// CIR:     %[[REF_TMP0:.*]] = cir.alloca !rec_G, !cir.ptr<!rec_G>, ["ref.tmp0"]
// CIR:     %[[REF_TMP1:.*]] = cir.alloca !rec_G, !cir.ptr<!rec_G>, ["ref.tmp1"]
// CIR:     %[[CLEANUP_TMP:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["tmp.exprcleanup"]
// CIR:     %[[LOAD_OBJ:.*]] = cir.load{{.*}} %[[OBJ]] : !cir.ptr<!cir.ptr<!rec_G>>, !cir.ptr<!rec_G>
// CIR:     %[[COPY:.*]] = cir.call @_ZNK1G4copyEv(%[[LOAD_OBJ]]) : (!cir.ptr<!rec_G> {{.*}}) -> !rec_G
// CIR:     cir.store{{.*}} %[[COPY]], %[[REF_TMP0]]
// CIR:     cir.cleanup.scope {
// CIR:       %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR:       cir.call @_ZN1GC1Ei(%[[REF_TMP1]], %[[ONE]]) : (!cir.ptr<!rec_G> {{.*}}, !s32i {{.*}}) -> ()
// CIR:       cir.cleanup.scope {
// CIR:         %[[EQUAL:.*]] = cir.call @_ZNK1GeqERKS_(%[[REF_TMP0]], %[[REF_TMP1]]) : (!cir.ptr<!rec_G> {{.*}}, !cir.ptr<!rec_G> {{.*}}) -> (!cir.bool {{.*}})
// CIR:         cir.store{{.*}} %[[EQUAL]], %[[CLEANUP_TMP]]
// CIR:         cir.yield
// CIR:       } cleanup  normal {
// CIR:         cir.call @_ZN1GD1Ev(%[[REF_TMP1]]) nothrow : (!cir.ptr<!rec_G> {{.*}}) -> ()
// CIR:         cir.yield
// CIR:       }
// CIR:       cir.yield
// CIR:     } cleanup  normal {
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
// LLVM:   %[[COPY:.*]] = call %struct.G @_ZNK1G4copyEv(ptr {{.*}} %[[LOAD_OBJ]])
// LLVM:   store %struct.G %[[COPY]], ptr %[[REF_TMP0]]
// LLVM:   br label %[[CLEAN_SCOPE_ONE:.*]]
// LLVM: [[CLEAN_SCOPE_ONE]]:
// LLVM:   call void @_ZN1GC1Ei(ptr {{.*}} %[[REF_TMP1]], i32 {{.*}} 1)
// LLVM:   br label %[[CLEAN_SCOPE_TWO:.*]]
// LLVM: [[CLEAN_SCOPE_TWO]]:
// LLVM:   %[[EQUAL:.*]] = call noundef i1 @_ZNK1GeqERKS_(ptr {{.*}} %[[REF_TMP0]], ptr {{.*}} %[[REF_TMP1]])
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

// Derived D2 (base) destructor -- does not call VirtualBase destructor

// CIR:     cir.func {{.*}} @_ZN7DerivedD2Ev
// CIR-NOT:   cir.call{{.*}} @_ZN11VirtualBaseD2Ev
// CIR:       cir.return

// LLVM:     define {{.*}} void @_ZN7DerivedD2Ev
// LLVM-NOT:   call{{.*}} @_ZN11VirtualBaseD2Ev
// LLVM:       ret

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

// OGCG emits these destructors in reverse order

// OGCG: define {{.*}} void @_ZN7DerivedD1Ev
// OGCG:   call void @_ZN7DerivedD2Ev(ptr {{.*}} %{{.*}}, ptr {{.*}} @_ZTT7Derived)
// OGCG:   call void @_ZN11VirtualBaseD2Ev(ptr {{.*}} %{{.*}})

// OGCG:     define {{.*}} void @_ZN7DerivedD2Ev
// OGCG-NOT:   call{{.*}} @_ZN11VirtualBaseD2Ev
// OGCG:       ret
