//===--- Function.h - Utility callable wrappers  -----------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides an analogue to std::function that supports move semantics.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_FUNCTION_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_FUNCTION_H

#include "llvm/ADT/STLExtras.h"
#include <cassert>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

namespace clang {
namespace clangd {

/// A move-only type-erasing function wrapper. Similar to `std::function`, but
/// allows to store move-only callables.
template <class> class UniqueFunction;

template <class Ret, class... Args> class UniqueFunction<Ret(Args...)> {
public:
  UniqueFunction() = default;
  UniqueFunction(std::nullptr_t) : UniqueFunction(){};

  UniqueFunction(UniqueFunction const &) = delete;
  UniqueFunction &operator=(UniqueFunction const &) = delete;

  UniqueFunction(UniqueFunction &&) noexcept = default;
  UniqueFunction &operator=(UniqueFunction &&) noexcept = default;

  template <class Callable,
            /// A sfinae-check that Callable can be called with Args... and
            class = typename std::enable_if<std::is_convertible<
                decltype(std::declval<Callable>()(std::declval<Args>()...)),
                Ret>::value>::type>
  UniqueFunction(Callable &&Func)
      : CallablePtr(llvm::make_unique<
                    FunctionCallImpl<typename std::decay<Callable>::type>>(
            std::forward<Callable>(Func))) {}

  explicit operator bool() { return bool(CallablePtr); }

  Ret operator()(Args... As) {
    assert(CallablePtr);
    return CallablePtr->Call(std::forward<Args>(As)...);
  }

private:
  class FunctionCallBase {
  public:
    virtual ~FunctionCallBase() = default;
    virtual Ret Call(Args... As) = 0;
  };

  template <class Callable>
  class FunctionCallImpl final : public FunctionCallBase {
    static_assert(
        std::is_same<Callable, typename std::decay<Callable>::type>::value,
        "FunctionCallImpl must be instanstiated with std::decay'ed types");

  public:
    FunctionCallImpl(Callable Func) : Func(std::move(Func)) {}

    Ret Call(Args... As) override { return Func(std::forward<Args>(As)...); }

  private:
    Callable Func;
  };

  std::unique_ptr<FunctionCallBase> CallablePtr;
};

/// Stores a callable object (Func) and arguments (Args) and allows to call the
/// callable with provided arguments later using `operator ()`. The arguments
/// are std::forward'ed into the callable in the body of `operator()`. Therefore
/// `operator()` can only be called once, as some of the arguments could be
/// std::move'ed into the callable on first call.
template <class Func, class... Args> struct ForwardBinder {
  using Tuple = std::tuple<typename std::decay<Func>::type,
                           typename std::decay<Args>::type...>;
  Tuple FuncWithArguments;
#ifndef NDEBUG
  bool WasCalled = false;
#endif

public:
  ForwardBinder(Tuple FuncWithArguments)
      : FuncWithArguments(std::move(FuncWithArguments)) {}

private:
  template <std::size_t... Indexes, class... RestArgs>
  auto CallImpl(llvm::integer_sequence<std::size_t, Indexes...> Seq,
                RestArgs &&... Rest)
      -> decltype(std::get<0>(this->FuncWithArguments)(
          std::forward<Args>(std::get<Indexes + 1>(this->FuncWithArguments))...,
          std::forward<RestArgs>(Rest)...)) {
    return std::get<0>(this->FuncWithArguments)(
        std::forward<Args>(std::get<Indexes + 1>(this->FuncWithArguments))...,
        std::forward<RestArgs>(Rest)...);
  }

public:
  template <class... RestArgs>
  auto operator()(RestArgs &&... Rest)
      -> decltype(this->CallImpl(llvm::index_sequence_for<Args...>(),
                                 std::forward<RestArgs>(Rest)...)) {

#ifndef NDEBUG
    assert(!WasCalled && "Can only call result of Bind once.");
    WasCalled = true;
#endif
    return CallImpl(llvm::index_sequence_for<Args...>(),
                    std::forward<RestArgs>(Rest)...);
  }
};

/// Creates an object that stores a callable (\p F) and first arguments to the
/// callable (\p As) and allows to call \p F with \Args at a later point.
/// Similar to std::bind, but also works with move-only \p F and \p As.
///
/// The returned object must be called no more than once, as \p As are
/// std::forwarded'ed (therefore can be moved) into \p F during the call.
template <class Func, class... Args>
ForwardBinder<Func, Args...> Bind(Func F, Args &&... As) {
  return ForwardBinder<Func, Args...>(
      std::make_tuple(std::forward<Func>(F), std::forward<Args>(As)...));
}

} // namespace clangd
} // namespace clang

#endif
