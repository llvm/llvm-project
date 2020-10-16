// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcoroutines-ts -std=c++14 -emit-llvm %s -o - -disable-llvm-passes | FileCheck %s

#include "Inputs/coroutine.h"

using namespace std::experimental;

namespace std {

struct nothrow_t {};
constexpr nothrow_t nothrow = {};

} // end namespace std

// Required when get_return_object_on_allocation_failure() is defined by
// the promise.
void* operator new(__SIZE_TYPE__ __sz, const std::nothrow_t&) noexcept;
void  operator delete(void* __p, const std::nothrow_t&) noexcept;


template <class RetObject>
struct promise_type {
    RetObject get_return_object();
    suspend_always initial_suspend();
    suspend_never final_suspend() noexcept;
    void return_void();
    static void unhandled_exception();
};

struct coro {
  using promise_type = promise_type<coro>;
  coro(coro const&);
  struct Impl;
  Impl *impl;
};

// Verify that the NRVO is applied to the Gro object.
// CHECK-LABEL: define void @_Z1fi(%struct.coro* noalias sret(%struct.coro) align 8 %agg.result, i32 %0)
coro f(int) {
// CHECK: %call = call noalias nonnull i8* @_Znwm(
// CHECK-NEXT: br label %[[CoroInit:.*]]

// CHECK: {{.*}}[[CoroInit]]:
// CHECK: store i1 false, i1* %gro.active
// CHECK: call void @{{.*get_return_objectEv}}(%struct.coro* sret(%struct.coro) align 8 %agg.result
// CHECK-NEXT: store i1 true, i1* %gro.active
  co_return;
}


template <class RetObject>
struct promise_type_with_on_alloc_failure {
    static RetObject get_return_object_on_allocation_failure();
    RetObject get_return_object();
    suspend_always initial_suspend();
    suspend_never final_suspend() noexcept;
    void return_void();
    static void unhandled_exception();
};

struct coro_two {
  using promise_type = promise_type_with_on_alloc_failure<coro_two>;
  coro_two(coro_two const&);
  struct Impl;
  Impl *impl;
};

// Verify that the NRVO is applied to the Gro object.
// CHECK-LABEL: define void @_Z1hi(%struct.coro_two* noalias sret(%struct.coro_two) align 8 %agg.result, i32 %0)
 coro_two h(int) {

// CHECK: %call = call noalias i8* @_ZnwmRKSt9nothrow_t
// CHECK-NEXT: %[[CheckNull:.*]] = icmp ne i8* %call, null
// CHECK-NEXT: br i1 %[[CheckNull]], label %[[InitOnSuccess:.*]], label %[[InitOnFailure:.*]]

// CHECK: {{.*}}[[InitOnFailure]]:
// CHECK-NEXT: call void @{{.*get_return_object_on_allocation_failureEv}}(%struct.coro_two* sret(%struct.coro_two) align 8 %agg.result
// CHECK-NEXT: br label %[[RetLabel:.*]]

// CHECK: {{.*}}[[InitOnSuccess]]:
// CHECK: store i1 false, i1* %gro.active
// CHECK: call void @{{.*get_return_objectEv}}(%struct.coro_two* sret(%struct.coro_two) align 8 %agg.result
// CHECK-NEXT: store i1 true, i1* %gro.active

// CHECK: [[RetLabel]]:
// CHECK-NEXT: ret void
  co_return;
}
