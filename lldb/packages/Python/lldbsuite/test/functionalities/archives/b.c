//===-- b.c -----------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
static int __b_global = 2;

int b(int arg) {
    int result = arg + __b_global;
    return result;
}

int bb(int arg1) {
    int result2 = arg1 - __b_global;
    return result2;
}
