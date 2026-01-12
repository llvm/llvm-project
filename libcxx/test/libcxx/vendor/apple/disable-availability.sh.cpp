//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: stdlib=apple-libc++

// This test ensures that we retain a way to disable availability markup on Apple platforms
// in order to work around Clang bug https://llvm.org/PR134151.
//
// Once that bug has been fixed or once we've made changes to libc++'s use of availability
// that render that workaround unnecessary, the macro and this test can be removed.
//
// The test works by creating a final linked image that refers to a function marked with
// both an availability attribute and with _LIBCPP_HIDE_FROM_ABI. We then check that this
// generates a weak reference to the function -- without the bug, we'd expect a strong
// reference or no reference at all instead.

// First, test the test. Make sure that we do (incorrectly) produce a weak definition when we
// don't define _LIBCPP_DISABLE_AVAILABILITY. Otherwise, something may have changed in libc++
// and this test might not work anymore.
// RUN: %{cxx} %s %{flags} %{compile_flags} %{link_flags} -fvisibility=hidden -fvisibility-inlines-hidden -shared -o %t.1.dylib
// RUN: nm -m %t.1.dylib | c++filt | grep value > %t.1.symbols
// RUN: grep weak %t.1.symbols

// Now, make sure that 'weak' goes away when we define _LIBCPP_DISABLE_AVAILABILITY.
// In fact, all references to the function might go away, so we just check that we don't emit
// any weak reference.
// RUN: %{cxx} %s %{flags} %{compile_flags} %{link_flags} -fvisibility=hidden -fvisibility-inlines-hidden -D_LIBCPP_DISABLE_AVAILABILITY -shared -o %t.2.dylib
// RUN: nm -m %t.2.dylib | c++filt | grep value > %t.2.symbols
// RUN: not grep weak %t.2.symbols

#include <version>

template <class T>
struct optional {
  T val_;
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_INTRODUCED_IN_LLVM_18_ATTRIBUTE T value() const { return val_; }
};

using PMF = int (optional<int>::*)() const;
PMF f() { return &optional<int>::value; }
