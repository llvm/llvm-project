// Tests defination of get-return-object-invocation [dcl.fct.def.coroutine] (and CWG2563)
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -emit-llvm %s -o - -disable-llvm-passes | FileCheck %s

#include "Inputs/coroutine.h"

using namespace std;

extern "C" {
void wrong();
}

template<bool LValue>
struct Promise {
    Promise() = default;
    Promise(const Promise&) { wrong(); }
    Promise(Promise&&) { wrong(); }

    // Tests: decltype(auto) gro = promise.get_return_object();
    auto&& get_return_object() {
      if constexpr (LValue)
        return *this;
      else
        return static_cast<Promise&&>(*this);
    }
    std::suspend_never initial_suspend() const { return {}; }
    std::suspend_never final_suspend() const noexcept { return {}; }
    void return_void() const {}
    void unhandled_exception() const noexcept {}
};

template<bool LValue>
struct Handle {
    using promise_type = Promise<LValue>;

    Handle(promise_type& p) {
      if constexpr (!LValue)
        wrong();
    }
    Handle(promise_type&& p) {
      if constexpr (LValue)
        wrong();
    }
};

Handle<true> lvalue() {
  co_return;
}

Handle<false> rvalue() {
  co_return;
}

// CHECK-NOT: call void @wrong
