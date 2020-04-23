//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make sure that we provide the %{verify} substitution. We can only test
// this when the verify-support feature is enabled, and it's difficult to
// check that it's enabled when it should be, so we just trust that it is.

// REQUIRES: verify-support
// RUN: test -n "%{verify}"

// RUN: %{cxx} %s %{flags} %{compile_flags} -fsyntax-only %{verify}

// expected-no-diagnostics
