// REQUIRES: host-supports-jit, x86_64-linux

// To generate libdynamic-library-test.so :
// clang -xc++ -o libdynamic-library-test.so -fPIC -shared
//
// extern "C" {
//
// int ultimate_answer = 0;
// 
// int calculate_answer() {
//   ultimate_answer = 42;
//   return 5;
// }
//
// }

// RUN: cat %s | env LD_LIBRARY_PATH=%S/Inputs:$LD_LIBRARY_PATH clang-repl | FileCheck %s

extern "C" int printf(const char* format, ...);

extern "C" int ultimate_answer;
extern "C" int calculate_answer();

%lib libdynamic-library-test.so

printf("Return value: %d\n", calculate_answer());
// CHECK: Return value: 5

printf("Variable: %d\n", ultimate_answer);
// CHECK-NEXT: Variable: 42

%quit
