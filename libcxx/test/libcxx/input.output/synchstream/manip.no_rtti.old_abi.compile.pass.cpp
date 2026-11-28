//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20
// UNSUPPORTED: no-localization
// UNSUPPORTED: libcpp-has-no-experimental-syncstream
// UNSUPPORTED: libcpp-abi-syncbuf-manip-virtual

// ADDITIONAL_COMPILE_FLAGS: -fno-rtti

// Without RTTI, the manipulators are not provided.

#include <concepts>
#include <ostream>
#include <syncstream>
#include <utility>

// Selected only if std doesn't provide the manipulator: the conversion to any_ref ranks below any std overload.
struct fallback {};
struct any_ref {
  template <class T>
  any_ref(T&);
};
fallback emit_on_flush(any_ref);
fallback noemit_on_flush(any_ref);
fallback flush_emit(any_ref);

template <class Os>
concept HasEmitOnFlush = !std::same_as<decltype(emit_on_flush(std::declval<Os&>())), fallback>;
template <class Os>
concept HasNoemitOnFlush = !std::same_as<decltype(noemit_on_flush(std::declval<Os&>())), fallback>;
template <class Os>
concept HasFlushEmit = !std::same_as<decltype(flush_emit(std::declval<Os&>())), fallback>;

static_assert(!HasEmitOnFlush<std::ostream>);
static_assert(!HasNoemitOnFlush<std::ostream>);
static_assert(!HasFlushEmit<std::ostream>);
static_assert(!HasEmitOnFlush<std::osyncstream>);
static_assert(!HasNoemitOnFlush<std::osyncstream>);
static_assert(!HasFlushEmit<std::osyncstream>);
