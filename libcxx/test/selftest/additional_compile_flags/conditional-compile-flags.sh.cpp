//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test ensures that we can add compile flags that are conditional on Lit features.

// ADDITIONAL_COMPILE_FLAGS(some-defined-feature): -this-flag-should-be-added
// ADDITIONAL_COMPILE_FLAGS(some-undefined-feature): -this-flag-should-not-be-added
// RUN: echo "%{compile_flags}" | grep -e '-this-flag-should-be-added'
// RUN: echo "%{compile_flags}" | grep -v -e '-this-flag-should-not-be-added'
