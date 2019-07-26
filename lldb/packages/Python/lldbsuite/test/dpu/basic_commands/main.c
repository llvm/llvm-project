//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

int fct3(int val) {
  val += 2;
  val *= 2;
  return val;
}

int fct2(int val) {
  val += 3; // StepIn entry location
  val = fct3(val);
  val = fct3(val); // Step location 2
  val++;
  return val;
}

int fct1(int val) {
  val++;
  val = fct2(val);
  val *= 2;
  val = fct2(val); // Step location 1
  val = val + val * 4;
  return val;
}

int main(int argc, char const *argv[]) {
  argc++; // Breakpoint location 1
  int ret = fct1(argc); // Breakpoint location 2

  return ret;
}
