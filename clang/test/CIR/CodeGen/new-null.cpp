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
