//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  // Perform a null pointer access.
  int *const null_int_ptr = nullptr;
  *null_int_ptr = 0xDEAD;

  return 0;
}
