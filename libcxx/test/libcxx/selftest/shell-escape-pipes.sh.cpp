//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make sure manually escaped pipes are handled properly by the shell.
// Specifically, we want to make sure that if we escape a pipe after %{exec},
// it gets executed by the %{exec} substitution, as opposed to the result of
// the %{exec} substitution being piped into the following command.
//
// This is a bit tricky to test. To test this, we basically want to ensure
// that both sides of the pipe are executed inside %{exec}. When we're inside
// %{exec}, the one difference we can rely on is that we're in a temporary
// directory with all file dependencies satisfied, so that's what we use.

// RUN: touch %t.foobar
// RUN: %{exec} echo \| ls > %t.out
// RUN: grep -e ".foobar" %t.out
