// REQUIRES: host-supports-jit, system-linux
// UNSUPPORTED: target={{.*-(ps4|ps5)}}

// RUN: %clang -xc++ -o %T/libdynamic-library-test.so -fPIC -shared -DLIBRARY %S/Inputs/dynamic-library-test.cpp
// RUN: cat %s | env LD_LIBRARY_PATH=%T:$LD_LIBRARY_PATH clang-repl | FileCheck %s

#include <cstdio>

extern int ultimate_answer;
int calculate_answer();

%lib libdynamic-library-test.so

printf("Return value: %d\n", calculate_answer());
// CHECK: Return value: 5

printf("Variable: %d\n", ultimate_answer);
// CHECK-NEXT: Variable: 42

%quit
