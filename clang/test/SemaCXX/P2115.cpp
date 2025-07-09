// RUN: rm -rf %t
// RUN: split-file %s %t


// RUN: %clang -std=c++20 -fmodule-header %t/A.h -o %t/A.pcm
// RUN: %clang -std=c++20 -fmodule-header %t/B.h -o %t/B.pcm
// RUN: %clang -std=c++20 -fsyntax-only -fmodule-file=%t/A.pcm -fmodule-file=%t/B.pcm %t/main.cpp

//--- A.h
// expected-no-diagnostics
enum { A = 0 };

//--- B.h
// expected-no-diagnostics
enum { B = 1 };

//--- main.cpp
// expected-no-diagnostics
import "A.h";
import "B.h";
int main() {}
