//===-- cpu_model_test.c - Test __builtin_cpu_supports -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __builtin_cpu_supports for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

int main (void) {
  if(__builtin_cpu_supports("avx2"))
    return 4;
  else
    return 3;
}
