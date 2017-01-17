//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

class Empty {};

int main (int argc, char const *argv[]) {
  Empty e;
  Empty* ep = new Empty;
  return 0; // Break at this line
}
