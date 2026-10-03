//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// FIXME: Encode whether the library was built with exceptions and check the abilist for that case as well

// REQUIRES: target=x86_64-unknown-linux-gnu && !no-exceptions

// Check that the list of symbols exported from the dylib doesn't change unexpectedly

// FILE_DEPENDENCIES: expected.abilist

// RUN: nm --defined-only --extern-only %{lib-dir}/libc++.so.1.0 | cut -c 18- | sort > %t.actual.abilist
// RUN: diff expected.abilist %t.actual.abilist
