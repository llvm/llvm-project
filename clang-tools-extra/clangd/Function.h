//===--- Function.h - Utility callable wrappers  -----------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides utilities for callable objects.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_FUNCTION_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_FUNCTION_H

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/Support/Error.h"
#include <mutex>
#include <tuple>
#include <utility>

namespace clang {
namespace clangd {

/// A Callback<T> is a void function that accepts Expected<T>.
/// This is accepted by ClangdServer functions that logically return T.
template <typename T>
using Callback = llvm::unique_function<void(llvm::Expected<T>)>;

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

/// An Event<T> allows events of type T to be broadcast to listeners.
template <typename T> class Event {
public:
  // A Listener is the callback through which events are delivered.
  using Listener = std::function<void(const T &)>;

  // A subscription defines the scope of when a listener should receive events.
  // After destroying the subscription, no more events are received.
  class LLVM_NODISCARD Subscription {
    Event *Parent;
    unsigned ListenerID;

    Subscription(Event *Parent, unsigned ListenerID)
        : Parent(Parent), ListenerID(ListenerID) {}
    friend Event;

  public:
    Subscription() : Parent(nullptr) {}
    Subscription(Subscription &&Other) : Parent(nullptr) {
      *this = std::move(Other);
    }
    Subscription &operator=(Subscription &&Other) {
      // If *this is active, unsubscribe.
      if (Parent) {
        std::lock_guard<std::recursive_mutex>(Parent->ListenersMu);
        llvm::erase_if(Parent->Listeners,
                       [&](const std::pair<Listener, unsigned> &P) {
                         return P.second == ListenerID;
                       });
      }
      // Take over the other subscription, and mark it inactive.
      std::tie(Parent, ListenerID) = std::tie(Other.Parent, Other.ListenerID);
      Other.Parent = nullptr;
      return *this;
    }
    // Destroying a subscription may block if an event is being broadcast.
    ~Subscription() {
      if (Parent)
        *this = Subscription(); // Unsubscribe.
    }
  };

  // Adds a listener that will observe all future events until the returned
  // subscription is destroyed.
  // May block if an event is currently being broadcast.
  Subscription observe(Listener L) {
    std::lock_guard<std::recursive_mutex> Lock(ListenersMu);
    Listeners.push_back({std::move(L), ++ListenerCount});
    return Subscription(this, ListenerCount);
  }

  // Synchronously sends an event to all registered listeners.
  // Must not be called from a listener to this event.
  void broadcast(const T &V) {
    // FIXME: it would be nice to dynamically check non-reentrancy here.
    std::lock_guard<std::recursive_mutex> Lock(ListenersMu);
    for (const auto &L : Listeners)
      L.first(V);
  }

  ~Event() {
    std::lock_guard<std::recursive_mutex> Lock(ListenersMu);
    assert(Listeners.empty());
  }

private:
  static_assert(std::is_same<typename std::decay<T>::type, T>::value,
                "use a plain type: event values are always passed by const&");

  std::recursive_mutex ListenersMu;
  bool IsBroadcasting = false;
  std::vector<std::pair<Listener, unsigned>> Listeners;
  unsigned ListenerCount = 0;
};

} // namespace clangd
} // namespace clang

#endif
