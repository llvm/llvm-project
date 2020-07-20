// RUN: %clangxx_profgen -fcoverage-mapping -Wno-comment -o %t %s
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: llvm-profdata merge -o %t.profdata %t.profraw
// RUN: llvm-cov show %t -instr-profile %t.profdata -path-equivalence=/tmp,%S 2>&1 | FileCheck %s

int main() {                           // CHECK:  [[# @LINE]]| 1|int main() {
    /* comment */ int x = 0;           // CHECK:  [[# @LINE]]| 1|  /* comment */ int x = 0;
    int y = 0; /* comment */           // CHECK:  [[# @LINE]]| 1|  int y = 0; /* comment */
    int z = 0; // comment              // CHECK:  [[# @LINE]]| 1|  int z = 0; // comment
    // comment                         // CHECK:  [[# @LINE]]|  |  // comment
                                       // CHECK:  [[# @LINE]]|  |
    x = 0; /*                          // CHECK:  [[# @LINE]]| 1|  x = 0; /*
    comment                            // CHECK:  [[# @LINE]]|  |    comment
    */                                 // CHECK:  [[# @LINE]]|  |    */
                                       // CHECK:  [[# @LINE]]|  |
    /*                                 // CHECK:  [[# @LINE]]|  |    /*
    comment                            // CHECK:  [[# @LINE]]|  |    comment
    */ x = 0;                          // CHECK:  [[# @LINE]]| 1|    */ x = 0;
                                       // CHECK:  [[# @LINE]]|  |
    /* comment */                      // CHECK:  [[# @LINE]]|  |    /* comment */
    // comment                         // CHECK:  [[# @LINE]]|  |    // comment
    /* comment */                      // CHECK:  [[# @LINE]]|  |    /* comment */
    z =                                // CHECK:  [[# @LINE]]| 1|    z =
    x // comment                       // CHECK:  [[# @LINE]]| 1|    x // comment
    // comment                         // CHECK:  [[# @LINE]]|  |    // comment
    + /*                               // CHECK:  [[# @LINE]]| 1|    + /*
    comment                            // CHECK:  [[# @LINE]]|  |    comment
    */                                 // CHECK:  [[# @LINE]]|  |    */
    /*                                 // CHECK:  [[# @LINE]]|  |    /*
    comment                            // CHECK:  [[# @LINE]]|  |    comment
    */y;                               // CHECK:  [[# @LINE]]| 1|    */y;
                                       // CHECK:  [[# @LINE]]|  |
    // Comments inside directives.     // CHECK:  [[# @LINE]]|  |    // Comments inside directives.
    #if 0 //comment                    // CHECK:  [[# @LINE]]|  |    #if 0 //comment
    /* comment */ x = 0;               // CHECK:  [[# @LINE]]|  |    /* comment */ x = 0;
    y = 0; /* comment */               // CHECK:  [[# @LINE]]|  |    y = 0; /* comment */
    z = 0; // comment                  // CHECK:  [[# @LINE]]|  |    z = 0; // comment
    // comment                         // CHECK:  [[# @LINE]]|  |    // comment
                                       // CHECK:  [[# @LINE]]|  |
    x = 0; /*                          // CHECK:  [[# @LINE]]|  |    x = 0; /*
    comment                            // CHECK:  [[# @LINE]]|  |    comment
    */                                 // CHECK:  [[# @LINE]]|  |    */
                                       // CHECK:  [[# @LINE]]|  | 
    /*                                 // CHECK:  [[# @LINE]]|  |    /*
    comment                            // CHECK:  [[# @LINE]]|  |    comment
    */ x = 0;                          // CHECK:  [[# @LINE]]|  |    */ x = 0;
                                       // CHECK:  [[# @LINE]]|  |
    /* comment */                      // CHECK:  [[# @LINE]]|  |    /* comment */
    // comment                         // CHECK:  [[# @LINE]]|  |    // comment
    /* comment */                      // CHECK:  [[# @LINE]]|  |    /* comment */
    #endif // comment                  // CHECK:  [[# @LINE]]|  |    #endif // comment
    #if 1 // comment                   // CHECK:  [[# @LINE]]| 1|    #if 1 // comment
    /* comment */ x = 0;               // CHECK:  [[# @LINE]]| 1|    /* comment */ x = 0;
    y = 0; /* comment */               // CHECK:  [[# @LINE]]| 1|    y = 0; /* comment */
    z = 0; // comment                  // CHECK:  [[# @LINE]]| 1|    z = 0; // comment
    // comment                         // CHECK:  [[# @LINE]]|  |    // comment
                                       // CHECK:  [[# @LINE]]|  |
    x = 0; /*                          // CHECK:  [[# @LINE]]| 1|    x = 0; /*
    comment                            // CHECK:  [[# @LINE]]|  |    comment
    */                                 // CHECK:  [[# @LINE]]|  |    */
                                       // CHECK:  [[# @LINE]]|  |
    /*                                 // CHECK:  [[# @LINE]]|  |    /*
    comment                            // CHECK:  [[# @LINE]]|  |    comment
    */ x = 0;                          // CHECK:  [[# @LINE]]| 1|    */ x = 0;
                                       // CHECK:  [[# @LINE]]|  |
    /* comment */                      // CHECK:  [[# @LINE]]|  |    /* comment */
    // comment                         // CHECK:  [[# @LINE]]|  |    // comment
    /* comment */                      // CHECK:  [[# @LINE]]|  |    /* comment */
    #endif //comment                   // CHECK:  [[# @LINE]]| 1|    #endif //comment
    return 0;                          // CHECK:  [[# @LINE]]| 1|    return 0;
}                                      // CHECK:  [[# @LINE]]| 1|}
