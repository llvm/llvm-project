// Remove comments first.
// RUN: sed 's/[ \t]*\/\/.*//' %s > %t.stripped.cpp
// RUN: %clangxx_profgen -fcoverage-mapping -o %t %t.stripped.cpp
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: llvm-profdata merge -o %t.profdata %t.profraw
// RUN: llvm-cov show %t -instr-profile %t.profdata 2>&1 | FileCheck %s


int main() {                        // CHECK:       [[# @LINE]]| 1|int main() {
    int x = 0;                      // CHECK-NEXT:  [[# @LINE]]| 1|
                                    // CHECK-NEXT:  [[# @LINE]]|  |
    x = 1;                          // CHECK-NEXT:  [[# @LINE]]| 1|
    if (x)                          // CHECK-NEXT:  [[# @LINE]]| 1|
                                    // CHECK-NEXT:  [[# @LINE]]|  |
        x                           // CHECK-NEXT:  [[# @LINE]]| 1|
                                    // CHECK-NEXT:  [[# @LINE]]|  |
        =                           // CHECK-NEXT:  [[# @LINE]]| 1|
                                    // CHECK-NEXT:  [[# @LINE]]|  |
                                    // CHECK-NEXT:  [[# @LINE]]|  |
        0;                          // CHECK-NEXT:  [[# @LINE]]| 1|
                                    // CHECK-NEXT:  [[# @LINE]]|  |
    if (x)                          // CHECK-NEXT:  [[# @LINE]]| 1|
                                    // CHECK-NEXT:  [[# @LINE]]|  |
                                    // CHECK-NEXT:  [[# @LINE]]|  |
        x = 1;                      // CHECK-NEXT:  [[# @LINE]]| 0|
                                    // CHECK-NEXT:  [[# @LINE]]|  |
    #ifdef UNDEFINED                // CHECK-NEXT:  [[# @LINE]]|  |
                                    // CHECK-NEXT:  [[# @LINE]]|  |
    int y = 0;                      // CHECK-NEXT:  [[# @LINE]]|  |
                                    // CHECK-NEXT:  [[# @LINE]]|  |
    y = 1;                          // CHECK-NEXT:  [[# @LINE]]|  |
    if (y)                          // CHECK-NEXT:  [[# @LINE]]|  |
                                    // CHECK-NEXT:  [[# @LINE]]|  |
        y                           // CHECK-NEXT:  [[# @LINE]]|  |
                                    // CHECK-NEXT:  [[# @LINE]]|  |
        =                           // CHECK-NEXT:  [[# @LINE]]|  |
                                    // CHECK-NEXT:  [[# @LINE]]|  |
                                    // CHECK-NEXT:  [[# @LINE]]|  |
        0;                          // CHECK-NEXT:  [[# @LINE]]|  |
                                    // CHECK-NEXT:  [[# @LINE]]|  |
    #endif                          // CHECK-NEXT:  [[# @LINE]]|  |
                                    // CHECK-NEXT:  [[# @LINE]]|  |
    #define DEFINED 1               // CHECK-NEXT:  [[# @LINE]]| 1|
                                    // CHECK-NEXT:  [[# @LINE]]|  |
    #ifdef DEFINED                  // CHECK-NEXT:  [[# @LINE]]| 1|
                                    // CHECK-NEXT:  [[# @LINE]]|  |
    int y = 0;                      // CHECK-NEXT:  [[# @LINE]]| 1|
                                    // CHECK-NEXT:  [[# @LINE]]|  |
    y = 1;                          // CHECK-NEXT:  [[# @LINE]]| 1|
    if (y)                          // CHECK-NEXT:  [[# @LINE]]| 1|
                                    // CHECK-NEXT:  [[# @LINE]]|  |
        y                           // CHECK-NEXT:  [[# @LINE]]| 1|
                                    // CHECK-NEXT:  [[# @LINE]]|  |
        =                           // CHECK-NEXT:  [[# @LINE]]| 1|
                                    // CHECK-NEXT:  [[# @LINE]]|  |
                                    // CHECK-NEXT:  [[# @LINE]]|  |
        0;                          // CHECK-NEXT:  [[# @LINE]]| 1|
    #endif                          // CHECK-NEXT:  [[# @LINE]]| 1|
                                    // CHECK-NEXT:  [[# @LINE]]|  |
    return 0;                       // CHECK-NEXT:  [[# @LINE]]| 1|
}                                   // CHECK-NEXT:  [[# @LINE]]| 1|
