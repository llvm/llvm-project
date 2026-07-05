// RUN: split-file %s %t
// RUN: cd %t
// RUN: %clang_pgogen -c shr1.c -o shr1.o -Xclang -fprofile-instrument-path=default_1.profraw
// RUN: %clang_pgogen -shared shr1.o -o shr1.so -mxcoff-build-id=0x01
//
// RUN: %clangxx_pgogen -c shr2.cpp -o shr2.o -Xclang -fprofile-instrument-path=default_2.profraw
// RUN: %clangxx_pgogen -shared shr2.o -o shr2.so -mxcoff-build-id=0x02
//
// RUN: %clang_pgogen -c main.c -o main.o -Xclang -fprofile-instrument-path=default_main.profraw
// RUN: %clang_pgogen main.o -L%t shr1.so shr2.so -o a.out -mxcoff-build-id=0xFFFFFFFFFFFFFFFF
//
// RUN: %run ./a.out
//
// RUN: llvm-profdata show --binary-ids default_1.profraw | FileCheck %s --check-prefix=SHARED1
// RUN: llvm-profdata show --binary-ids default_2.profraw | FileCheck %s --check-prefix=SHARED2
// RUN: llvm-profdata show --binary-ids default_main.profraw | FileCheck %s --check-prefix=MAIN

// SHARED1: Binary IDs:
// SHARED1-NEXT:  {{^}}01{{$}}
// SHARED2: Binary IDs:
// SHARED2-NEXT:  {{^}}02{{$}}
// MAIN: Binary IDs:
// MAIN-NEXT:  {{^}}ffffffffffffffff{{$}}

//--- shr1.c
int shr1() { return 1; }

//--- shr2.cpp
int helper() { return 3; }
extern "C" int shr2() { return helper(); }

//--- main.c
int shr1();
int shr2();
int main() { return 4 - shr1() - shr2(); }
