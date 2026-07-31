// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fexceptions -fcxx-exceptions -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fexceptions -fcxx-exceptions -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fexceptions -fcxx-exceptions -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

typedef __typeof__(sizeof(int)) size_t;

namespace std {
  struct nothrow_t {};
}
std::nothrow_t nothrow;

void *operator new(size_t, const std::nothrow_t &) throw();
void operator delete(void *, const std::nothrow_t &) throw();

struct S {
  S();
  ~S();
  int a;
};

// nothrow new with non-POD type triggers null check
S *test_nothrow_new() {
  return new (nothrow) S;
}

// CHECK: cir.func {{.*}} @_Z16test_nothrow_newv()
// CHECK:   %[[ALLOC:.*]] = cir.call @_ZnwmRKSt9nothrow_t({{.*}}) nothrow
// CHECK:   %[[NULL:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!void>
// CHECK:   %[[IS_NOT_NULL:.*]] = cir.cmp ne %[[ALLOC]], %[[NULL]] : !cir.ptr<!void>
// CHECK:   cir.if %[[IS_NOT_NULL]] {
// CHECK:     cir.cleanup.scope {
// CHECK:       %[[CAST:.*]] = cir.cast bitcast %[[ALLOC]] : !cir.ptr<!void> -> !cir.ptr<!rec_S>
// CHECK:       cir.call @_ZN1SC1Ev(%[[CAST]])
// CHECK:     } cleanup eh {
// CHECK:       cir.call @_ZdlPvRKSt9nothrow_t(%[[ALLOC]], {{.*}}) nothrow
// CHECK:     } loc(
// CHECK:   } loc(
// CHECK-NEXT: %[[LOADED:.*]] = cir.load
// CHECK:   %[[NULL_S:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!rec_S>
// CHECK:   cir.select if %[[IS_NOT_NULL]] then %[[LOADED]] else %[[NULL_S]]

// LLVM: define {{.*}} ptr @_Z16test_nothrow_newv() {{.*}}personality ptr @__gxx_personality_v0
// LLVM:   %[[ALLOC:.*]] = call {{.*}} ptr @_ZnwmRKSt9nothrow_t(i64 noundef 4, {{.*}})
// LLVM:   %[[CMP:.*]] = icmp ne ptr %[[ALLOC]], null
// LLVM:   br i1 %[[CMP]], label %[[NOT_NULL:.*]], label %[[CONT:.*]]
// LLVM: [[NOT_NULL]]:
// LLVM:   invoke void @_ZN1SC1Ev({{.*}} %[[ALLOC]])
// LLVM:     to label {{.*}} unwind label %[[LPAD:.*]]
// LLVM: [[LPAD]]:
// LLVM:   landingpad { ptr, i32 }
// LLVM:     cleanup
// LLVM:   call void @_ZdlPvRKSt9nothrow_t({{.*}} %[[ALLOC]], {{.*}})
// LLVM:   resume
// LLVM: [[CONT]]:
// LLVM:   select i1 %[[CMP]], ptr {{.*}}, ptr null

// OGCG: define {{.*}} ptr @_Z16test_nothrow_newv() {{.*}}personality ptr @__gxx_personality_v0
// OGCG:   %[[ALLOC:.*]] = call {{.*}} ptr @_ZnwmRKSt9nothrow_t(i64 noundef 4, {{.*}})
// OGCG:   %[[IS_NULL:.*]] = icmp eq ptr %[[ALLOC]], null
// OGCG:   br i1 %[[IS_NULL]], label %[[CONT:.*]], label %[[NOT_NULL:.*]]
// OGCG: [[NOT_NULL]]:
// OGCG:   invoke void @_ZN1SC1Ev({{.*}} %[[ALLOC]])
// OGCG:     to label %[[OK:.*]] unwind label %[[LPAD:.*]]
// OGCG: [[OK]]:
// OGCG:   br label %[[CONT]]
// OGCG: [[CONT]]:
// OGCG:   phi ptr
// OGCG: [[LPAD]]:
// OGCG:   landingpad { ptr, i32 }
// OGCG:     cleanup
// OGCG:   call void @_ZdlPvRKSt9nothrow_t({{.*}} %[[ALLOC]], {{.*}})
// OGCG:   resume

// nothrow new with POD + initializer triggers null check
int *test_nothrow_new_init() {
  return new (nothrow) int(42);
}

// CHECK: cir.func {{.*}} @_Z21test_nothrow_new_initv()
// CHECK:   %[[ALLOC:.*]] = cir.call @_ZnwmRKSt9nothrow_t({{.*}}) nothrow
// CHECK:   %[[NULL:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!void>
// CHECK:   %[[IS_NOT_NULL:.*]] = cir.cmp ne %[[ALLOC]], %[[NULL]] : !cir.ptr<!void>
// CHECK:   cir.if %[[IS_NOT_NULL]] {
// CHECK:     cir.cleanup.scope {
// CHECK:       %[[CAST:.*]] = cir.cast bitcast %[[ALLOC]] : !cir.ptr<!void> -> !cir.ptr<!s32i>
// CHECK:       %[[FORTY_TWO:.*]] = cir.const #cir.int<42> : !s32i
// CHECK:       cir.store {{.*}} %[[FORTY_TWO]], %[[CAST]]
// CHECK:     } cleanup eh {
// CHECK:       cir.call @_ZdlPvRKSt9nothrow_t(%[[ALLOC]], {{.*}}) nothrow
// CHECK:     } loc(
// CHECK:   } loc(
// CHECK-NEXT: %[[LOADED_I:.*]] = cir.load
// CHECK:   %[[NULL_I:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!s32i>
// CHECK:   cir.select if %[[IS_NOT_NULL]] then %[[LOADED_I]] else %[[NULL_I]]

// LLVM: define {{.*}} ptr @_Z21test_nothrow_new_initv()
// LLVM:   %[[ALLOC:.*]] = call {{.*}} ptr @_ZnwmRKSt9nothrow_t(i64 noundef 4, {{.*}})
// LLVM:   %[[CMP:.*]] = icmp ne ptr %[[ALLOC]], null
// LLVM:   br i1 %[[CMP]], label %[[NOT_NULL:.*]], label %[[CONT:.*]]
// LLVM: [[NOT_NULL]]:
// LLVM:   store i32 42, ptr %[[ALLOC]], align 4
// LLVM: [[CONT]]:
// LLVM:   select i1 %[[CMP]], ptr {{.*}}, ptr null

// OGCG: define {{.*}} ptr @_Z21test_nothrow_new_initv()
// OGCG:   %[[ALLOC:.*]] = call {{.*}} ptr @_ZnwmRKSt9nothrow_t(i64 noundef 4, {{.*}})
// OGCG:   %[[IS_NULL:.*]] = icmp eq ptr %[[ALLOC]], null
// OGCG:   br i1 %[[IS_NULL]], label %[[CONT:.*]], label %[[NOT_NULL:.*]]
// OGCG: [[NOT_NULL]]:
// OGCG:   store i32 42, ptr %[[ALLOC]], align 4
// OGCG:   br label %[[CONT]]
// OGCG: [[CONT]]:
// OGCG:   phi ptr

struct T {
  T();
  ~T();
  operator int();
};

T makeT();

struct U {
  U(int);
  ~U();
};

// nothrow new with a temporary in the initializer: the temporary's dtor cleanup
// must be conditional because the initializer only runs when allocation succeeds.
// The operator delete cleanup stays unconditional: it is entered entirely inside
// the null-check branch.
U *test_nothrow_new_temp() {
  return new (nothrow) U(makeT());
}

// CHECK: cir.func {{.*}} @_Z21test_nothrow_new_tempv()
// CHECK:   %[[TMP:.*]] = cir.alloca "ref.tmp0" {{.*}} : !cir.ptr<!rec_T>
// CHECK:   %[[TMP_ACTIVE:.*]] = cir.alloca "cleanup.cond" {{.*}} : !cir.ptr<!cir.bool>
// CHECK:   cir.cleanup.scope {
// CHECK:     %[[ALLOC:.*]] = cir.call @_ZnwmRKSt9nothrow_t({{.*}}) nothrow
// CHECK:     %[[NULL:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!void>
// CHECK:     %[[IS_NOT_NULL:.*]] = cir.cmp ne %[[ALLOC]], %[[NULL]] : !cir.ptr<!void>
// CHECK:     %[[FALSE:.*]] = cir.const #false
// CHECK:     cir.store %[[FALSE]], %[[TMP_ACTIVE]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK:     cir.if %[[IS_NOT_NULL]] {
// CHECK:       cir.cleanup.scope {
// CHECK:         %[[MAKE_T:.*]] = cir.call @_Z5makeTv() : () -> !rec_T
// CHECK:         cir.store{{.*}} %[[MAKE_T]], %[[TMP]] : !rec_T, !cir.ptr<!rec_T>
// CHECK:         %[[TRUE:.*]] = cir.const #true
// CHECK:         cir.store %[[TRUE]], %[[TMP_ACTIVE]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK:         %[[CONV:.*]] = cir.call @_ZN1TcviEv(%[[TMP]])
// CHECK:         cir.call @_ZN1UC1Ei({{.*}}, %[[CONV]])
// CHECK:       } cleanup eh {
// CHECK:         cir.call @_ZdlPvRKSt9nothrow_t(%[[ALLOC]], {{.*}}) nothrow
// CHECK:       } loc({{.*}})
// CHECK:     } loc({{.*}})
// CHECK:     %[[LOADED:.*]] = cir.load{{.*}} : !cir.ptr<!cir.ptr<!rec_U>>, !cir.ptr<!rec_U>
// CHECK:     %[[NULL_U:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!rec_U>
// CHECK:     cir.select if %[[IS_NOT_NULL]] then %[[LOADED]] else %[[NULL_U]]
// CHECK:   } cleanup all {
// CHECK:     %[[TMP_IS_ACTIVE:.*]] = cir.load{{.*}} %[[TMP_ACTIVE]] : !cir.ptr<!cir.bool>, !cir.bool
// CHECK:     cir.if %[[TMP_IS_ACTIVE]] {
// CHECK:       cir.call @_ZN1TD1Ev(%[[TMP]]) nothrow
// CHECK:     }
// CHECK:   }

// LLVM: define {{.*}} ptr @_Z21test_nothrow_new_tempv() {{.*}}personality ptr @__gxx_personality_v0
// LLVM:   %[[TMP:.*]] = alloca %struct.T
// LLVM:   %[[ALLOC:.*]] = call {{.*}} ptr @_ZnwmRKSt9nothrow_t(i64 noundef 1, {{.*}})
// LLVM:   %[[CMP:.*]] = icmp ne ptr %[[ALLOC]], null
// LLVM:   store i8 0, ptr %[[TMP_ACTIVE:.*]]
// LLVM:   br i1 %[[CMP]], label %[[NOT_NULL:.*]], label %[[CONT:.*]]
// LLVM: [[NOT_NULL]]:
// LLVM:   %[[MAKE_T:.*]] = invoke %struct.T @_Z5makeTv()
// LLVM:           to label %[[INVOKE_CONT:.*]] unwind label %[[LPAD:.*]]
// LLVM: [[INVOKE_CONT]]:
// LLVM:   store {{.*}} %[[MAKE_T]], ptr %[[TMP]]
// LLVM:   store i8 1, ptr %[[TMP_ACTIVE]]
// LLVM:   invoke {{.*}} @_ZN1TcviEv(ptr {{.*}} %[[TMP]])
// LLVM:   invoke void @_ZN1UC1Ei(ptr {{.*}} %[[ALLOC]], i32 {{.*}})
// LLVM: [[LPAD]]:
// LLVM:   landingpad { ptr, i32 }
// LLVM:          cleanup
// LLVM:   call void @_ZdlPvRKSt9nothrow_t({{.*}} %[[ALLOC]], {{.*}})
// LLVM: [[CONT]]:
// LLVM:   select i1 %[[CMP]], ptr {{.*}}, ptr null
// LLVM:   %[[TMP_I8:.*]] = load i8, ptr %[[TMP_ACTIVE]]
// LLVM:   %[[TMP_IS_ACTIVE:.*]] = trunc i8 %[[TMP_I8]] to i1
// LLVM:   br i1 %[[TMP_IS_ACTIVE]], label %[[DO_TMP_DTOR:.*]], label %[[SKIP_TMP_DTOR:.*]]
// LLVM: [[DO_TMP_DTOR]]:
// LLVM:   call void @_ZN1TD1Ev(ptr {{.*}} %[[TMP]])

// OGCG: define {{.*}} ptr @_Z21test_nothrow_new_tempv() {{.*}}personality ptr @__gxx_personality_v0
// OGCG: entry:
// OGCG:   %[[TMP:.*]] = alloca %struct.T
// OGCG:   %[[ALLOC:.*]] = call {{.*}} ptr @_ZnwmRKSt9nothrow_t(i64 noundef 1, {{.*}})
// OGCG:   %[[IS_NULL:.*]] = icmp eq ptr %[[ALLOC]], null
// OGCG:   store i1 false, ptr %[[DELETE_ACTIVE:.*]]
// OGCG:   store i1 false, ptr %[[TMP_ACTIVE:.*]]
// OGCG:   br i1 %[[IS_NULL]], label %[[CONT:.*]], label %[[NOT_NULL:.*]]
// OGCG: [[NOT_NULL]]:
// OGCG:   store i1 true, ptr %[[DELETE_ACTIVE]]
// OGCG:   invoke void @_Z5makeTv(ptr {{.*}} %[[TMP]])
// OGCG:           to label %[[INVOKE_CONT:.*]] unwind label %[[LPAD:.*]]
// OGCG: [[INVOKE_CONT]]:
// OGCG:   store i1 true, ptr %[[TMP_ACTIVE]]
// OGCG:   invoke {{.*}} @_ZN1TcviEv(ptr {{.*}} %[[TMP]])
// OGCG:   invoke void @_ZN1UC1Ei(ptr {{.*}} %[[ALLOC]], i32 {{.*}})
// OGCG:   store i1 false, ptr %[[DELETE_ACTIVE]]
// OGCG:   br label %[[CONT]]
// OGCG: [[CONT]]:
// OGCG:   phi ptr
// OGCG:   %[[TMP_IS_ACTIVE:.*]] = load i1, ptr %[[TMP_ACTIVE]]
// OGCG:   br i1 %[[TMP_IS_ACTIVE]], label %[[DO_TMP_DTOR:.*]], label %[[SKIP_TMP_DTOR:.*]]
// OGCG: [[DO_TMP_DTOR]]:
// OGCG:   call void @_ZN1TD1Ev(ptr {{.*}} %[[TMP]])
// OGCG: [[LPAD]]:
// OGCG:   landingpad { ptr, i32 }
// OGCG:          cleanup
// OGCG:   %[[DEL_ACTIVE:.*]] = load i1, ptr %[[DELETE_ACTIVE]]
// OGCG:   br i1 %[[DEL_ACTIVE]], label %[[DO_DELETE:.*]], label %[[SKIP_DELETE:.*]]
// OGCG: [[DO_DELETE]]:
// OGCG:   call void @_ZdlPvRKSt9nothrow_t({{.*}})

struct InnerT {
  InnerT();
  ~InnerT();
  operator int();
};

struct OuterT {
  OuterT(int);
  ~OuterT();
  operator int();
};

InnerT makeInnerT();
OuterT makeOuterT(int);

// Nested temporaries in a nothrow-new initializer: each temporary gets its own
// conditional cleanup flag, destroyed in reverse construction order.
U *test_nothrow_new_nested_temps() {
  return new (nothrow) U(makeOuterT(makeInnerT()));
}

// CHECK: cir.func {{.*}} @_Z29test_nothrow_new_nested_tempsv()
// CHECK:   %[[OUTER_TMP:.*]] = cir.alloca "ref.tmp0" {{.*}} : !cir.ptr<!rec_OuterT>
// CHECK:   %[[INNER_TMP:.*]] = cir.alloca "ref.tmp1" {{.*}} : !cir.ptr<!rec_InnerT>
// CHECK:   %[[INNER_ACTIVE:.*]] = cir.alloca "cleanup.cond" {{.*}} : !cir.ptr<!cir.bool>
// CHECK:   %[[OUTER_ACTIVE:.*]] = cir.alloca "cleanup.cond" {{.*}} : !cir.ptr<!cir.bool>
// CHECK:   cir.cleanup.scope {
// CHECK:     %[[ALLOC:.*]] = cir.call @_ZnwmRKSt9nothrow_t({{.*}}) nothrow
// CHECK:     %[[NULL:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!void>
// CHECK:     %[[IS_NOT_NULL:.*]] = cir.cmp ne %[[ALLOC]], %[[NULL]] : !cir.ptr<!void>
// CHECK:     %[[FALSE:.*]] = cir.const #false
// CHECK:     cir.store %[[FALSE]], %[[INNER_ACTIVE]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK:     %[[FALSE2:.*]] = cir.const #false
// CHECK:     cir.store %[[FALSE2]], %[[OUTER_ACTIVE]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK:     cir.if %[[IS_NOT_NULL]] {
// CHECK:       cir.cleanup.scope {
// CHECK:         %[[MAKE_INNER:.*]] = cir.call @_Z10makeInnerTv() : () -> !rec_InnerT
// CHECK:         cir.store{{.*}} %[[MAKE_INNER]], %[[INNER_TMP]] : !rec_InnerT, !cir.ptr<!rec_InnerT>
// CHECK:         %[[TRUE:.*]] = cir.const #true
// CHECK:         cir.store %[[TRUE]], %[[INNER_ACTIVE]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK:         %[[INNER_CONV:.*]] = cir.call @_ZN6InnerTcviEv(%[[INNER_TMP]])
// CHECK:         %[[MAKE_OUTER:.*]] = cir.call @_Z10makeOuterTi(%[[INNER_CONV]]) : (!s32i {{.*}}) -> !rec_OuterT
// CHECK:         cir.store{{.*}} %[[MAKE_OUTER]], %[[OUTER_TMP]] : !rec_OuterT, !cir.ptr<!rec_OuterT>
// CHECK:         %[[TRUE2:.*]] = cir.const #true
// CHECK:         cir.store %[[TRUE2]], %[[OUTER_ACTIVE]] : !cir.bool, !cir.ptr<!cir.bool>
// CHECK:         %[[OUTER_CONV:.*]] = cir.call @_ZN6OuterTcviEv(%[[OUTER_TMP]])
// CHECK:         cir.call @_ZN1UC1Ei({{.*}}, %[[OUTER_CONV]])
// CHECK:       } cleanup eh {
// CHECK:         cir.call @_ZdlPvRKSt9nothrow_t(%[[ALLOC]], {{.*}}) nothrow
// CHECK:       } loc({{.*}})
// CHECK:     } loc({{.*}})
// CHECK:     %[[LOADED:.*]] = cir.load{{.*}} : !cir.ptr<!cir.ptr<!rec_U>>, !cir.ptr<!rec_U>
// CHECK:     %[[NULL_U:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!rec_U>
// CHECK:     cir.select if %[[IS_NOT_NULL]] then %[[LOADED]] else %[[NULL_U]]
// CHECK:   } cleanup all {
// CHECK:     %[[OUTER_IS_ACTIVE:.*]] = cir.load{{.*}} %[[OUTER_ACTIVE]] : !cir.ptr<!cir.bool>, !cir.bool
// CHECK:     cir.if %[[OUTER_IS_ACTIVE]] {
// CHECK:       cir.call @_ZN6OuterTD1Ev(%[[OUTER_TMP]]) nothrow
// CHECK:     }
// CHECK:     %[[INNER_IS_ACTIVE:.*]] = cir.load{{.*}} %[[INNER_ACTIVE]] : !cir.ptr<!cir.bool>, !cir.bool
// CHECK:     cir.if %[[INNER_IS_ACTIVE]] {
// CHECK:       cir.call @_ZN6InnerTD1Ev(%[[INNER_TMP]]) nothrow
// CHECK:     }
// CHECK:   }

// LLVM: define {{.*}} ptr @_Z29test_nothrow_new_nested_tempsv() {{.*}}personality ptr @__gxx_personality_v0
// LLVM:   %[[OUTER_TMP:.*]] = alloca %struct.OuterT
// LLVM:   %[[INNER_TMP:.*]] = alloca %struct.InnerT
// LLVM:   %[[ALLOC:.*]] = call {{.*}} ptr @_ZnwmRKSt9nothrow_t(i64 noundef 1, {{.*}})
// LLVM:   %[[CMP:.*]] = icmp ne ptr %[[ALLOC]], null
// LLVM:   store i8 0, ptr %[[INNER_ACTIVE:.*]]
// LLVM:   store i8 0, ptr %[[OUTER_ACTIVE:.*]]
// LLVM:   br i1 %[[CMP]], label %[[NOT_NULL:.*]], label %[[CONT:.*]]
// LLVM: [[NOT_NULL]]:
// LLVM:   %[[MAKE_INNER:.*]] = invoke %struct.InnerT @_Z10makeInnerTv()
// LLVM:           to label %[[INNER_CONT:.*]] unwind label %[[LPAD:.*]]
// LLVM: [[INNER_CONT]]:
// LLVM:   store {{.*}} %[[MAKE_INNER]], ptr %[[INNER_TMP]]
// LLVM:   store i8 1, ptr %[[INNER_ACTIVE]]
// LLVM:   %[[INNER_CONV:.*]] = invoke {{.*}} @_ZN6InnerTcviEv(ptr {{.*}} %[[INNER_TMP]])
// LLVM:           to label %[[INNER_CONV_CONT:.*]] unwind label %[[LPAD]]
// LLVM: [[INNER_CONV_CONT]]:
// LLVM:   %[[MAKE_OUTER:.*]] = invoke %struct.OuterT @_Z10makeOuterTi(i32 {{.*}} %[[INNER_CONV]])
// LLVM:           to label %[[OUTER_CONT:.*]] unwind label %[[LPAD]]
// LLVM: [[OUTER_CONT]]:
// LLVM:   store {{.*}} %[[MAKE_OUTER]], ptr %[[OUTER_TMP]]
// LLVM:   store i8 1, ptr %[[OUTER_ACTIVE]]
// LLVM:   invoke {{.*}} @_ZN6OuterTcviEv(ptr {{.*}} %[[OUTER_TMP]])
// LLVM:   invoke void @_ZN1UC1Ei(ptr {{.*}} %[[ALLOC]], i32 {{.*}})
// LLVM: [[LPAD]]:
// LLVM:   landingpad { ptr, i32 }
// LLVM:          cleanup
// LLVM:   call void @_ZdlPvRKSt9nothrow_t({{.*}} %[[ALLOC]], {{.*}})
// LLVM: [[CONT]]:
// LLVM:   select i1 %[[CMP]], ptr {{.*}}, ptr null
// LLVM:   %[[OUTER_I8:.*]] = load i8, ptr %[[OUTER_ACTIVE]]
// LLVM:   %[[OUTER_IS_ACTIVE:.*]] = trunc i8 %[[OUTER_I8]] to i1
// LLVM:   br i1 %[[OUTER_IS_ACTIVE]], label %[[DO_OUTER_DTOR:.*]], label %[[SKIP_OUTER_DTOR:.*]]
// LLVM: [[DO_OUTER_DTOR]]:
// LLVM:   call void @_ZN6OuterTD1Ev(ptr {{.*}} %[[OUTER_TMP]])
// LLVM: [[SKIP_OUTER_DTOR]]:
// LLVM:   %[[INNER_I8:.*]] = load i8, ptr %[[INNER_ACTIVE]]
// LLVM:   %[[INNER_IS_ACTIVE:.*]] = trunc i8 %[[INNER_I8]] to i1
// LLVM:   br i1 %[[INNER_IS_ACTIVE]], label %[[DO_INNER_DTOR:.*]], label %[[SKIP_INNER_DTOR:.*]]
// LLVM: [[DO_INNER_DTOR]]:
// LLVM:   call void @_ZN6InnerTD1Ev(ptr {{.*}} %[[INNER_TMP]])

// OGCG: define {{.*}} ptr @_Z29test_nothrow_new_nested_tempsv() {{.*}}personality ptr @__gxx_personality_v0
// OGCG: entry:
// OGCG:   %[[OUTER_TMP:.*]] = alloca %struct.OuterT
// OGCG:   %[[INNER_TMP:.*]] = alloca %struct.InnerT
// OGCG:   %[[ALLOC:.*]] = call {{.*}} ptr @_ZnwmRKSt9nothrow_t(i64 noundef 1, {{.*}})
// OGCG:   %[[IS_NULL:.*]] = icmp eq ptr %[[ALLOC]], null
// OGCG:   store i1 false, ptr %[[DELETE_ACTIVE:.*]]
// OGCG:   store i1 false, ptr %[[INNER_ACTIVE:.*]]
// OGCG:   store i1 false, ptr %[[OUTER_ACTIVE:.*]]
// OGCG:   br i1 %[[IS_NULL]], label %[[CONT:.*]], label %[[NOT_NULL:.*]]
// OGCG: [[NOT_NULL]]:
// OGCG:   store i1 true, ptr %[[DELETE_ACTIVE]]
// OGCG:   invoke void @_Z10makeInnerTv(ptr {{.*}} %[[INNER_TMP]])
// OGCG:           to label %[[INNER_CONT:.*]] unwind label %[[LPAD:.*]]
// OGCG: [[INNER_CONT]]:
// OGCG:   store i1 true, ptr %[[INNER_ACTIVE]]
// OGCG:   %[[INNER_CONV:.*]] = invoke {{.*}} @_ZN6InnerTcviEv(ptr {{.*}} %[[INNER_TMP]])
// OGCG:           to label %[[INNER_CONV_CONT:.*]] unwind label %[[LPAD_INNER:.*]]
// OGCG: [[INNER_CONV_CONT]]:
// OGCG:   invoke void @_Z10makeOuterTi(ptr {{.*}} %[[OUTER_TMP]], i32 {{.*}} %[[INNER_CONV]])
// OGCG:           to label %[[OUTER_CONT:.*]] unwind label %[[LPAD_INNER]]
// OGCG: [[OUTER_CONT]]:
// OGCG:   store i1 true, ptr %[[OUTER_ACTIVE]]
// OGCG:   invoke {{.*}} @_ZN6OuterTcviEv(ptr {{.*}} %[[OUTER_TMP]])
// OGCG:   invoke void @_ZN1UC1Ei(ptr {{.*}} %[[ALLOC]], i32 {{.*}})
// OGCG:   store i1 false, ptr %[[DELETE_ACTIVE]]
// OGCG:   br label %[[CONT]]
// OGCG: [[CONT]]:
// OGCG:   phi ptr
// OGCG:   %[[OUTER_IS_ACTIVE:.*]] = load i1, ptr %[[OUTER_ACTIVE]]
// OGCG:   br i1 %[[OUTER_IS_ACTIVE]], label %[[DO_OUTER_DTOR:.*]], label %[[SKIP_OUTER_DTOR:.*]]
// OGCG: [[DO_OUTER_DTOR]]:
// OGCG:   call void @_ZN6OuterTD1Ev(ptr {{.*}} %[[OUTER_TMP]])
// OGCG: [[SKIP_OUTER_DTOR]]:
// OGCG:   %[[INNER_IS_ACTIVE:.*]] = load i1, ptr %[[INNER_ACTIVE]]
// OGCG:   br i1 %[[INNER_IS_ACTIVE]], label %[[DO_INNER_DTOR:.*]], label %[[SKIP_INNER_DTOR:.*]]
// OGCG: [[DO_INNER_DTOR]]:
// OGCG:   call void @_ZN6InnerTD1Ev(ptr {{.*}} %[[INNER_TMP]])
// OGCG: [[LPAD]]:
// OGCG:   landingpad { ptr, i32 }
// OGCG:          cleanup
// OGCG:   %[[DEL_ACTIVE:.*]] = load i1, ptr %[[DELETE_ACTIVE]]
// OGCG:   br i1 %[[DEL_ACTIVE]], label %[[DO_DELETE:.*]], label %[[SKIP_DELETE:.*]]
// OGCG: [[DO_DELETE]]:
// OGCG:   call void @_ZdlPvRKSt9nothrow_t({{.*}})
