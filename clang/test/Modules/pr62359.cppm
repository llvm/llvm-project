// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/Hello.cppm -o %t/Hello.pcm
// RUN: not %clang_cc1 -std=c++20 -fopenmp %t/use.cpp -fmodule-file=hello=%t/Hello.pcm -fsyntax-only \
// RUN:     2>&1 | FileCheck %t/use.cpp
// RUN: not %clang_cc1 -std=c++20 -fopenmp %t/use2.cpp -fmodule-file=hello=%t/Hello.pcm -fsyntax-only \
// RUN:     2>&1 | FileCheck %t/use2.cpp
//
// RUN: %clang_cc1 -std=c++20 -fopenmp -emit-module-interface %t/Hello.cppm -o %t/Hello.pcm
// RUN: %clang_cc1 -std=c++20 -fopenmp %t/use.cpp -fmodule-file=hello=%t/Hello.pcm -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 -fopenmp %t/use2.cpp -fmodule-file=hello=%t/Hello.pcm -fsyntax-only -verify

// Test again with reduced BMI
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface %t/Hello.cppm -o %t/Hello.pcm
// RUN: not %clang_cc1 -std=c++20 -fopenmp %t/use.cpp -fmodule-file=hello=%t/Hello.pcm -fsyntax-only \
// RUN:     2>&1 | FileCheck %t/use.cpp
// RUN: not %clang_cc1 -std=c++20 -fopenmp %t/use2.cpp -fmodule-file=hello=%t/Hello.pcm -fsyntax-only \
// RUN:     2>&1 | FileCheck %t/use2.cpp
//
// RUN: %clang_cc1 -std=c++20 -fopenmp -emit-reduced-module-interface %t/Hello.cppm -o %t/Hello.pcm
// RUN: %clang_cc1 -std=c++20 -fopenmp %t/use.cpp -fmodule-file=hello=%t/Hello.pcm -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 -fopenmp %t/use2.cpp -fmodule-file=hello=%t/Hello.pcm -fsyntax-only -verify


//--- Hello.cppm
export module hello;
export void hello() {
  
}

//--- use.cpp
// expected-no-diagnostics
import hello;
int use() {
  for(int i=0;i<10;i++)
    hello();
  return 0;
}

// CHECK: OpenMP{{.*}}differs in AST file '{{.*}}Hello.pcm' vs. current file

//--- use2.cpp
// expected-no-diagnostics
import hello;
int use2() {
#pragma omp parallel for
  for(int i=0;i<10;i++)
    hello();
  return 0;
}

// CHECK: OpenMP{{.*}}differs in AST file '{{.*}}Hello.pcm' vs. current file
// CHECK: use of undeclared identifier 'pragma'
