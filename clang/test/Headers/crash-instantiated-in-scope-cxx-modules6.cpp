// RUN: rm -fR %t
// RUN: split-file %s %t
// RUN: cd %t
// RUN: %clang_cc1 -std=c++20 -xc++ -emit-module -fmodules foo.cppmap -fmodule-name=foo -fmodule-name=foo -o foo.pcm
// RUN: %clang_cc1 -std=c++20 -xc++ -emit-module -fmodules bar.cppmap -fmodule-name=bar -o bar.pcm
// RUN: %clang_cc1 -std=c++20 -xc++ -emit-module -fmodules experiment_context.cppmap -fmodule-name=experiment_context -fmodule-file=foo.pcm -fmodule-file=bar.pcm -o experiment_context.pcm
// RUN: %clang_cc1 -verify -std=c++20 -xc++ -fmodule-file=experiment_context.pcm experiment_context_test.cc -o experiment_context_test.o

// https://github.com/llvm/llvm-project/issues/141582
//--- bar.cppmap
module "bar" {
  export *
  header "co.h"
}

//--- foo.cppmap
module "foo" {
  export *
  header "co.h"
}

//--- experiment_context.cppmap
module "experiment_context" {
  export *
  header "lazy.h"

  use "foo"
  use "bar"
}

//--- experiment_context_test.cc
// expected-no-diagnostics
#include "lazy.h"

namespace c9 {

template <typename T>
void WaitForCoroutine() {
  MakeCo<T>([]() -> Co<void> {
    co_return;
  });
}

void test() {
  c9::WaitForCoroutine<void>();
}
}

//--- lazy.h
#pragma once

#include "co.h"

namespace c9 {
template <typename T, typename F>
Co<T> MakeCo(F f)
{
  co_return co_await f();
}
}

inline c9::Co<void> DoNothing() { co_return; }


//--- co.h
#pragma  once
namespace std {

template <class _Ret, class... _Args>
struct coroutine_traits {};

template <typename Ret, typename... Args>
  requires requires { typename Ret::promise_type; }
struct coroutine_traits<Ret, Args...> {
  using promise_type = typename Ret::promise_type;
};

template <typename Promise = void>
struct coroutine_handle;

template <>
struct coroutine_handle<void> {};

template <typename Promise = void>
struct coroutine_handle : coroutine_handle<> {
  static coroutine_handle from_address(void *addr);
};

struct suspend_always {
  bool await_ready() noexcept;
  void await_suspend(coroutine_handle<>) noexcept;
  void await_resume() noexcept;
};

}  // namespace std

namespace c9 {

template <typename T>
class Co;

namespace internal {

template <typename T>
class CoroutinePromise {
 public:
  template <typename... Args>
  explicit CoroutinePromise(Args&&... args) {
    // Ensure that the 'dummy_color' VarDecl referenced by the inner DeclRefExpr
    // is the same declaration as the one outside the lambda.
    // This is guaranteed because both CoroutinePromise and the lambda's call operator
    // (CXXMethodDecl) are loaded from the same module.
    const int dummy_color = 1;
    [&]{ (void)dummy_color; }();
  }

  ~CoroutinePromise();
  void return_void();
  auto get_return_object() {
    return Co<T>();
  }
  void unhandled_exception();
  std::suspend_always  initial_suspend();

  struct result_t {
    ~result_t();
    bool await_ready() const noexcept;
    void await_suspend(std::coroutine_handle<void>) noexcept;
    void await_resume() const noexcept;
  };

  template <typename U>
  result_t await_transform(Co<U> co);

  std::suspend_always final_suspend() noexcept;
};
}  // namespace internal

template <typename T>
class Co {
 public:
  using promise_type = internal::CoroutinePromise<void>;
};

class CoIncomingModuleBase {
 public:
    Co<void> CoAfterFinish() { co_return; }
};
}  // namespace c9
