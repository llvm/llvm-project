// RUN: %check_clang_tidy %s readability-enum-initial-value %t

enum class EError {
  // CHECK-MESSAGES: :[[@LINE-1]]:1:  warning: initial values in enum 'EError' are not consistent
  EError_a = 1,
  EError_b,
  // CHECK-FIXES: EError_b = 2,
  EError_c = 3,
};

enum class ENone {
  ENone_a,
  ENone_b,
  EENone_c,
};

enum class EFirst {
  EFirst_a = 1,
  EFirst_b,
  EFirst_c,
};

enum class EAll {
  EAll_a = 1,
  EAll_b = 2,
  EAll_c = 3,
};
