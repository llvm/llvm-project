// RUN: %check_clang_tidy %s readability-enum-initial-value %t

enum EError {
  // CHECK-MESSAGES: :[[@LINE-1]]:1: warning: inital value in enum EError has readability issue, explicit initialization of all of enumerators
  EError_a = 1,
  EError_b,
  // CHECK-FIXES: EError_b = 2,
  EError_c = 3,
};

enum ENone {
  ENone_a,
  ENone_b,
  eENone_c,
};

enum EFirst {
  EFirst_a = 1,
  EFirst_b,
  EFirst_c,
};

enum EAll {
  EAll_a = 1,
  EAll_b = 2,
  EAll_c = 3,
};
