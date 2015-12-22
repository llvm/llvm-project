//===-- main.cpp --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
typedef int Foo;

int main() {
    Foo array[3] = {1,2,3};
    return 0; //% self.expect("frame variable array --show-types --", substrs = ['(Foo [3]) wrong_type_here = {','(Foo) [0] = 1','(Foo) [1] = 2','(Foo) [2] = 3'])
}
