//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

class PrintfContainer {
public:
    int printf() {
        return 0;
    }
};

int main() {
    PrintfContainer().printf(); //% self.expect("expression -- printf(\"Hello\\n\")", substrs = ['6'])
    return 0;
}

