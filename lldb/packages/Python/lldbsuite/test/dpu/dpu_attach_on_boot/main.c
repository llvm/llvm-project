//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

__attribute__((noinline)) int fct3(int val) {
  val += 2;
  val *= 2;
  return val;
}

__attribute__((noinline)) int fct2(int val) {
  val += 3; // StepIn entry location
  int val2 = fct3(val);
  int val3 = fct3(val2); // Step location 2
  val3++;
  return val3;
}

__attribute__((noinline)) int fct1(int val) {
  val++;
  int val2 = fct2(val);
  val2 *= 2;
  int val3 = fct2(val2); // Step location 1
  val3 = val3 + val3 * 4;
  return val3;
}

__attribute__((noinline)) int main(int argc, char const *argv[]) {
  argc++; // Breakpoint location 1
  int ret = fct1(argc); // Breakpoint location 2

  return ret;
}
