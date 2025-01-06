// XFAIL: target={{.*}}-aix{{.*}}
// RUN: %clangxx_profgen -fcoverage-mapping -Wno-comment -o %t %s
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: llvm-profdata merge -o %t.profdata %t.profraw
// RUN: llvm-cov show %t -instr-profile %t.profdata 2>&1 | FileCheck %s

int main() {                           // CHECK:       [[# @LINE]]| 1|int main() {
    /* comment */ int x = 0;           // CHECK-NEXT:  [[# @LINE]]| 1|
    int y = 0; /* comment */           // CHECK-NEXT:  [[# @LINE]]| 1|
    int z = 0; // comment              // CHECK-NEXT:  [[# @LINE]]| 1|
    // comment                         // CHECK-NEXT:  [[# @LINE]]|  |
                                       // CHECK-NEXT:  [[# @LINE]]|  |
    x = 0; /*                          // CHECK-NEXT:  [[# @LINE]]| 1|
    comment                            // CHECK-NEXT:  [[# @LINE]]|  |
    */                                 // CHECK-NEXT:  [[# @LINE]]|  |
                                       // CHECK-NEXT:  [[# @LINE]]|  |
    /*                                 // CHECK-NEXT:  [[# @LINE]]|  |
    comment                            // CHECK-NEXT:  [[# @LINE]]|  |
    */ x = 0;                          // CHECK-NEXT:  [[# @LINE]]| 1|
                                       // CHECK-NEXT:  [[# @LINE]]|  |
    /* comment */                      // CHECK-NEXT:  [[# @LINE]]|  |
    // comment                         // CHECK-NEXT:  [[# @LINE]]|  |
    /* comment */                      // CHECK-NEXT:  [[# @LINE]]|  |
    z =                                // CHECK-NEXT:  [[# @LINE]]| 1|
    x // comment                       // CHECK-NEXT:  [[# @LINE]]| 1|
    // comment                         // CHECK-NEXT:  [[# @LINE]]|  |
    + /*                               // CHECK-NEXT:  [[# @LINE]]| 1|
    comment                            // CHECK-NEXT:  [[# @LINE]]|  |
    */                                 // CHECK-NEXT:  [[# @LINE]]|  |
    /*                                 // CHECK-NEXT:  [[# @LINE]]|  |
    comment                            // CHECK-NEXT:  [[# @LINE]]|  |
    */y;                               // CHECK-NEXT:  [[# @LINE]]| 1|
                                       // CHECK-NEXT:  [[# @LINE]]|  |
    // Comments inside directives.     // CHECK-NEXT:  [[# @LINE]]|  |
    #if 0 //comment                    // CHECK-NEXT:  [[# @LINE]]|  |
    /* comment */ x = 0;               // CHECK-NEXT:  [[# @LINE]]|  |
    y = 0; /* comment */               // CHECK-NEXT:  [[# @LINE]]|  |
    z = 0; // comment                  // CHECK-NEXT:  [[# @LINE]]|  |
    // comment                         // CHECK-NEXT:  [[# @LINE]]|  |
                                       // CHECK-NEXT:  [[# @LINE]]|  |
    x = 0; /*                          // CHECK-NEXT:  [[# @LINE]]|  |
    comment                            // CHECK-NEXT:  [[# @LINE]]|  |
    */                                 // CHECK-NEXT:  [[# @LINE]]|  |
                                       // CHECK-NEXT:  [[# @LINE]]|  | 
    /*                                 // CHECK-NEXT:  [[# @LINE]]|  |
    comment                            // CHECK-NEXT:  [[# @LINE]]|  |
    */ x = 0;                          // CHECK-NEXT:  [[# @LINE]]|  |
                                       // CHECK-NEXT:  [[# @LINE]]|  |
    /* comment */                      // CHECK-NEXT:  [[# @LINE]]|  |
    // comment                         // CHECK-NEXT:  [[# @LINE]]|  |
    /* comment */                      // CHECK-NEXT:  [[# @LINE]]|  |
    #endif // comment                  // CHECK-NEXT:  [[# @LINE]]|  |
    #if 1 // comment                   // CHECK-NEXT:  [[# @LINE]]| 1|
    /* comment */ x = 0;               // CHECK-NEXT:  [[# @LINE]]| 1|
    y = 0; /* comment */               // CHECK-NEXT:  [[# @LINE]]| 1|
    z = 0; // comment                  // CHECK-NEXT:  [[# @LINE]]| 1|
    // comment                         // CHECK-NEXT:  [[# @LINE]]|  |
                                       // CHECK-NEXT:  [[# @LINE]]|  |
    x = 0; /*                          // CHECK-NEXT:  [[# @LINE]]| 1|
    comment                            // CHECK-NEXT:  [[# @LINE]]|  |
    */                                 // CHECK-NEXT:  [[# @LINE]]|  |
                                       // CHECK-NEXT:  [[# @LINE]]|  |
    /*                                 // CHECK-NEXT:  [[# @LINE]]|  |
    comment                            // CHECK-NEXT:  [[# @LINE]]|  |
    */ x = 0;                          // CHECK-NEXT:  [[# @LINE]]| 1|
                                       // CHECK-NEXT:  [[# @LINE]]|  |
    /* comment */                      // CHECK-NEXT:  [[# @LINE]]|  |
    // comment                         // CHECK-NEXT:  [[# @LINE]]|  |
    /* comment */                      // CHECK-NEXT:  [[# @LINE]]|  |
    #endif //comment                   // CHECK-NEXT:  [[# @LINE]]| 1|
    return 0;                          // CHECK-NEXT:  [[# @LINE]]| 1|
}                                      // CHECK-NEXT:  [[# @LINE]]| 1|
