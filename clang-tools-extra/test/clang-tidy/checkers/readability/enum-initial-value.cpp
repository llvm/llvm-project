// RUN: %check_clang_tidy %s readability-enum-initial-value %t

enum class EError {
  // CHECK-MESSAGES: :[[@LINE-1]]:1: warning: initial values in enum 'EError' are not consistent, consider explicit initialization of all, none or only the first enumerator
  EError_a = 1,
  EError_b,
  // CHECK-MESSAGES: :[[@LINE-1]]:3: note: uninitialized enumerator 'EError_b' defined here
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
