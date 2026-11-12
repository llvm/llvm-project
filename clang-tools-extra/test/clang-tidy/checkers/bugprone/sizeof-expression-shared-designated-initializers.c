// RUN: %check_clang_tidy -std=c99 %s bugprone-sizeof-expression %t-c -- -- -pedantic-errors
// RUN: %check_clang_tidy -std=c++20 %s bugprone-sizeof-expression %t-cxx -- -- -x c++ -pedantic-errors

// Nested designated initializers share subtrees without compound literals in
// both C and C++. The single sizeof warning must not be repeated per AST path.
struct V0 { unsigned long value; };
struct V1 { struct V0 child; };
struct V2 { struct V1 child; };
struct V3 { struct V2 child; };
struct V4 { struct V3 child; };
struct V5 { struct V4 child; };
struct V6 { struct V5 child; };
struct V7 { struct V6 child; };
struct V8 { struct V7 child; };
struct V9 { struct V8 child; };
struct V10 { struct V9 child; };
struct V11 { struct V10 child; };
struct V12 { struct V11 child; };
struct V13 { struct V12 child; };
struct V14 { struct V13 child; };
struct V15 { struct V14 child; };
struct V16 { struct V15 child; };
struct V17 { struct V16 child; };
struct V18 { struct V17 child; };
struct V19 { struct V18 child; };
struct V20 { struct V19 child; };

void f(void) {
  struct V20 value = {
    .child = {
      .child = {
        .child = {
          .child = {
            .child = {
              .child = {
                .child = {
                  .child = {
                    .child = {
                      .child = {
                        .child = {
                          .child = {
                            .child = {
                              .child = {
                                .child = {
                                  .child = {
                                    .child = {
                                      .child = {
                                        .child = {
                                          .child = {
                                            .value = sizeof(sizeof(int))
// CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: suspicious usage of 'sizeof(sizeof(...))' [bugprone-sizeof-expression]
                                          }
                                        }
                                      }
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  };
  (void)value;
}
