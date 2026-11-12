// RUN: %check_clang_tidy %s modernize-use-default-member-init %t -- \
// RUN:   -config="{CheckOptions: {modernize-use-default-member-init.IgnoreMacros: false}}"

#define MACRO() \
  struct S { \
    void *P; \
    S() : P(nullptr) {} \
  };

MACRO();
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use default member initializer for 'P'

struct S2 {
  void *P;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use default member initializer for 'P'
  S2() : P(nullptr) {}
};
