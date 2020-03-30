//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make sure that %{file_dependencies} is empty when no FILE_DEPENDENCIES
// line appears. Amongst other things, this makes sure that we don't share
// file dependencies across unrelated Lit tests.

// RUN: test -z "%{file_dependencies}"
