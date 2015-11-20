//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

int main() {
  union {
    int i;
    char c;
  };
  struct {
    int x;
    char y;
    short z;
  } s{3,'B',14};
  i = 0xFFFFFF00;
  c = 'A';
  return c; // break here
}
