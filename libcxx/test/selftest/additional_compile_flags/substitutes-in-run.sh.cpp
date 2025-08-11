//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make sure that additional compiler flags are added to the %{compile_flags}
// substitution.

// ADDITIONAL_COMPILE_FLAGS: -foo
// ADDITIONAL_COMPILE_FLAGS: -bar
// ADDITIONAL_COMPILE_FLAGS: -baz -foom
// RUN: echo "%{compile_flags}" | grep -e '-foo -bar -baz -foom'
