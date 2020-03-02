// UNSUPPORTED: *
// 1) Compile shared code into different object files and into an executable.

// RUN: %clangxx_profgen -std=c++14 -fcoverage-mapping %s -c -o %t.v1.o \
// RUN:                  -D_VERSION_1
// RUN: %clangxx_profgen -std=c++14 -fcoverage-mapping %s -c -o %t.v2.o \
// RUN:                  -D_VERSION_2
// RUN: %clangxx_profgen -std=c++14 -fcoverage-mapping %t.v1.o %t.v2.o \
// RUN:                  -o %t.exe

// 2) Collect profile data.

// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t.exe
// RUN: llvm-profdata merge %t.profraw -o %t.profdata

// 3) Generate coverage reports from the different object files and the exe.

// RUN: llvm-cov show %t.v1.o -instr-profile=%t.profdata | FileCheck %s -check-prefixes=V1,V1-ONLY
// RUN: llvm-cov show %t.v2.o -instr-profile=%t.profdata | FileCheck %s -check-prefixes=V2,V2-ONLY
// RUN: llvm-cov show %t.v1.o -object %t.v2.o -instr-profile=%t.profdata | FileCheck %s -check-prefixes=V1,V2
// RUN: llvm-cov show %t.exe -instr-profile=%t.profdata | FileCheck %s -check-prefixes=V1,V2

// 4) Verify that coverage reporting on the aggregate coverage mapping shows
//    hits for all code. (We used to arbitrarily pick a mapping from one binary
//    and prefer it over others.) When only limited coverage information is
//    available (just from one binary), don't try to guess any region counts.

struct A {
  A() {}    // V1: [[@LINE]]{{ *}}|{{ *}}1
            // V1-ONLY: [[@LINE+1]]{{ *}}|{{ *}}|
  A(int) {} // V2-ONLY: [[@LINE-2]]{{ *}}|{{ *}}|
            // V2: [[@LINE-1]]{{ *}}|{{ *}}1
};

#ifdef _VERSION_1

void foo();

void bar() {
  A x;      // V1: [[@LINE]]{{ *}}|{{ *}}1
}

int main() {
  foo();    // V1: [[@LINE]]{{ *}}|{{ *}}1
  bar();
  return 0;
}

#endif // _VERSION_1

#ifdef _VERSION_2

void foo() {
  A x{0};   // V2: [[@LINE]]{{ *}}|{{ *}}1
}

#endif // _VERSION_2
