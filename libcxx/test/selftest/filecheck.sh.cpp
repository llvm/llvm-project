//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-filecheck

// Make sure that we can use filecheck to write tests when the `has-filecheck`
// Lit feature is defined.

// RUN: echo "hello world" | filecheck %s
// CHECK: hello world
