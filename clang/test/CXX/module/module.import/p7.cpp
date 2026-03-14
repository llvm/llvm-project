// RUN: mkdir -p %t
// RUN: split-file %s %t

// All of the following should build without diagnostics.
//
// RUN: %clang_cc1 -std=c++20 %t/a.cpp  -emit-module-interface -o %t/a.pcm
// R U N: %clang_cc1 -std=c++20 %t/a.pcm  -emit-obj -o %t/a.o
//
// RUN: %clang_cc1 -std=c++20 %t/b.cpp  -emit-module-interface -o %t/b.pcm \
// RUN: -fprebuilt-module-path=%t 
// R U N: %clang_cc1 -std=c++20 %t/b.pcm  -emit-obj -o %t/b.o
//
// RUN: %clang_cc1 -std=c++20 %t/b-impl.cpp -emit-obj -o %t/b-impl.o \
// RUN: -fprebuilt-module-path=%t
//
// RUN: %clang_cc1 -std=c++20 %t/ab-main.cpp  -fsyntax-only \
// RUN: -fprebuilt-module-path=%t

//--- a.cpp

export module a;

export int foo() {
   return 42;
}

//--- b.cpp

export module b;
import a;

export int bar();

//--- b-impl.cpp

module b;

int bar() {
   return foo();
}

//--- ab-main.cpp

import b;

int main() {
   return bar();
}

