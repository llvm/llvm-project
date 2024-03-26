// RUN: %check_clang_tidy %s readability-enum-initial-value %t

enum EError {
  // CHECK-MESSAGES: :[[@LINE-1]]:1: warning: inital values in enum 'EError' are not consistent
  EError_a = 1,
  EError_b,
  // CHECK-FIXES: EError_b = 2,
  EError_c = 3,
};

enum ENone {
  ENone_a,
  ENone_b,
  EENone_c,
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

#define ENUMERATOR_1 EMacro1_b
enum EMacro1 {
  // CHECK-MESSAGES: :[[@LINE-1]]:1: warning: inital values in enum 'EMacro1' are not consistent
  EMacro1_a = 1,
  ENUMERATOR_1,
  EMacro1_c = 3,
};


#define ENUMERATOR_2 EMacro2_b = 2
enum EMacro2 {
  // CHECK-MESSAGES: :[[@LINE-1]]:1: warning: inital values in enum 'EMacro2' are not consistent
  EMacro2_a = 1,
  ENUMERATOR_2,
  EMacro2_c,
};
