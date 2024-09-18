// RUN: rm -fR %t
// RUN: split-file %s %t
// RUN: cd %t
// RUN: %clang_cc1 -std=c++20 -emit-header-unit -xc++-user-header -Werror=uninitialized folly-conv.h
// RUN: %clang_cc1 -std=c++20 -emit-header-unit -xc++-user-header -Werror=uninitialized thrift_cpp2_base.h
// RUN: %clang_cc1 -std=c++20 -emit-header-unit -xc++-user-header -Werror=uninitialized -fmodule-file=folly-conv.pcm -fmodule-file=thrift_cpp2_base.pcm logger_base.h

//--- Conv.h
#pragma once

template <typename _Tp, typename _Up = _Tp&&>
_Up __declval(int);

template <typename _Tp>
auto declval() noexcept -> decltype(__declval<_Tp>(0));

namespace folly {

template <class Value, class Error>
struct Expected {
  template <class Yes>
  auto thenOrThrow() -> decltype(declval<Value&>()) {
    return 1;
  }
};

struct ExpectedHelper {
  template <class Error, class T>
  static constexpr Expected<T, Error> return_(T) {
    return Expected<T, Error>();
  }

  template <class This, class Fn, class E = int, class T = ExpectedHelper>
  static auto then_(This&&, Fn&&)
      -> decltype(T::template return_<E>((declval<Fn>()(true), 0))) {
    return Expected<int, int>();
  }
};

template <class Tgt>
inline Expected<Tgt, const char*> tryTo() {
  Tgt result = 0;
  // In build with asserts:
  // clang/lib/Sema/SemaTemplateInstantiate.cpp: llvm::PointerUnion<Decl *, LocalInstantiationScope::DeclArgumentPack *> *clang::LocalInstantiationScope::findInstantiationOf(const Decl *): Assertion `isa<LabelDecl>(D) && "declaration not instantiated in this scope"' failed.
  // In release build compilation error on the line below inside lambda:
  // error: variable 'result' is uninitialized when used here [-Werror,-Wuninitialized]
  ExpectedHelper::then_(Expected<bool, int>(), [&](bool) { return result; });
  return {};
}

} // namespace folly

inline void bar() {
  folly::tryTo<int>();
}
// expected-no-diagnostics

//--- folly-conv.h
#pragma once
#include "Conv.h"
// expected-no-diagnostics

//--- thrift_cpp2_base.h
#pragma once
#include "Conv.h"
// expected-no-diagnostics

//--- logger_base.h
#pragma once
import "folly-conv.h";
import "thrift_cpp2_base.h";

inline void foo() {
  folly::tryTo<unsigned>();
}
// expected-no-diagnostics
