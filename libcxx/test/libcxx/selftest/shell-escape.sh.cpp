//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make sure that arguments of the %{exec} substitution are shell-escaped
// properly. If that wasn't the case, the command would fail because the
// shell would look for a matching `"`.

// RUN: %{exec} echo '"'

// Also make sure that we don't escape Shell builtins like `!`, because the
// shell otherwise thinks it's a command and it can't find it.

// RUN: ! false
