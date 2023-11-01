//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: verify-support

// Make sure that we provide the %{verify} convenience substitution.

// RUN: %{verify}

static_assert(false, ""); // expected-error {{static assertion failed}}
