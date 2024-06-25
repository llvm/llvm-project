// RUN: %check_clang_tidy %s readability-enum-initial-value %t
// RUN: %check_clang_tidy -check-suffix=ENABLE %s readability-enum-initial-value %t -- \
// RUN:     -config='{CheckOptions: { \
// RUN:         readability-enum-initial-value.AllowExplicitZeroFirstInitialValue: false, \
// RUN:         readability-enum-initial-value.AllowExplicitSequentialInitialValues: false, \
// RUN:     }}'

enum EError {
  // CHECK-MESSAGES: :[[@LINE-1]]:1: warning: inital values in enum 'EError' are not consistent
  // CHECK-MESSAGES-ENABLE: :[[@LINE-2]]:1: warning: inital values in enum 'EError' are not consistent
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
  EAll_c = 4,
};

#define ENUMERATOR_1 EMacro1_b
enum EMacro1 {
  // CHECK-MESSAGES: :[[@LINE-1]]:1: warning: inital values in enum 'EMacro1' are not consistent
  // CHECK-MESSAGES-ENABLE: :[[@LINE-2]]:1: warning: inital values in enum 'EMacro1' are not consistent
  EMacro1_a = 1,
  ENUMERATOR_1,
  // CHECK-FIXES: ENUMERATOR_1 = 2,
  EMacro1_c = 3,
};


#define ENUMERATOR_2 EMacro2_b = 2
enum EMacro2 {
  // CHECK-MESSAGES: :[[@LINE-1]]:1: warning: inital values in enum 'EMacro2' are not consistent
  // CHECK-MESSAGES-ENABLE: :[[@LINE-2]]:1: warning: inital values in enum 'EMacro2' are not consistent
  EMacro2_a = 1,
  ENUMERATOR_2,
  EMacro2_c,
  // CHECK-FIXES: EMacro2_c = 3,
};

enum EnumZeroFirstInitialValue {
  EnumZeroFirstInitialValue_0 = 0,
  // CHECK-MESSAGES-ENABLE: :[[@LINE-1]]:3: warning: zero initial value for the first enumerator in 'EnumZeroFirstInitialValue' can be disregarded
  // CHECK-FIXES-ENABLE: EnumZeroFirstInitialValue_0 ,
  EnumZeroFirstInitialValue_1,
  EnumZeroFirstInitialValue_2,
};

enum EnumZeroFirstInitialValueWithComment {
  EnumZeroFirstInitialValueWithComment_0 = /* == */ 0,
  // CHECK-MESSAGES-ENABLE: :[[@LINE-1]]:3: warning: zero initial value for the first enumerator in 'EnumZeroFirstInitialValueWithComment' can be disregarded
  // CHECK-FIXES-ENABLE: EnumZeroFirstInitialValueWithComment_0 /* == */ ,
  EnumZeroFirstInitialValueWithComment_1,
  EnumZeroFirstInitialValueWithComment_2,
};

enum EnumSequentialInitialValue {
  // CHECK-MESSAGES-ENABLE: :[[@LINE-1]]:1: warning: sequential initial value in 'EnumSequentialInitialValue' can be ignored
  EnumSequentialInitialValue_0 = 2,
  // CHECK-FIXES-ENABLE: EnumSequentialInitialValue_0 = 2,
  EnumSequentialInitialValue_1 = 3,
  // CHECK-FIXES-ENABLE: EnumSequentialInitialValue_1 ,
  EnumSequentialInitialValue_2 = 4,
  // CHECK-FIXES-ENABLE: EnumSequentialInitialValue_2 ,
};
