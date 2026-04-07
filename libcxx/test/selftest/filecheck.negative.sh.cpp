//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-filecheck

// Make sure that FileCheck fails when it should fail. This ensure that FileCheck
// actually checks the content of the file.
// XFAIL: *

// RUN: echo "hello world" | FileCheck %s
// CHECK: foobar
