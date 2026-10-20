//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make sure we can mark a gen-test as UNSUPPORTED

// We use C++03 as a random feature that we know exists. The goal is to make
// this test always unsupported.
// UNSUPPORTED: c++03
// REQUIRES: c++03

// Note that an unsupported gen-test should still contain some commands, otherwise
// what are we generating? They are never executed, though.
// RUN: something-definitely-invalid
