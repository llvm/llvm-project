//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

class foo {
public:
  template <class T> T func(T x) const {
    return x+2; //% self.expect("expr 2+3", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["5"])
  }
};

int i;

int main() {
  return foo().func(i);
}
