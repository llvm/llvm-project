// RUN: %check_clang_tidy %s cppcoreguidelines-avoid-non-const-global-variables %t -- \
// RUN: -config="{CheckOptions: {cppcoreguidelines-avoid-non-const-global-variables.IgnoreMacros: false}}"
// RUN: %check_clang_tidy %s -check-suffixes=IGNORE-MACROS cppcoreguidelines-avoid-non-const-global-variables %t -- \
// RUN: -config="{CheckOptions: {cppcoreguidelines-avoid-non-const-global-variables.IgnoreMacros: true}}"

int nonConstInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: variable 'nonConstInt' is non-const and globally accessible, consider making it const
// CHECK-MESSAGES-IGNORE-MACROS: :[[@LINE-2]]:5: warning: variable 'nonConstInt' is non-const and globally accessible, consider making it const

int &nonConstReference = nonConstInt;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: variable 'nonConstReference' provides global access to a non-const object; consider making the referenced data 'const'
// CHECK-MESSAGES-IGNORE-MACROS: :[[@LINE-2]]:6: warning: variable 'nonConstReference' provides global access to a non-const object; consider making the referenced data 'const'

#define DEFINE_NON_CONST_INT int macroInt = 0;

DEFINE_NON_CONST_INT
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: variable 'macroInt' is non-const and globally accessible, consider making it const

#define DEFINE_NON_CONST_REFERENCE int &macroRef = nonConstInt;

DEFINE_NON_CONST_REFERENCE
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: variable 'macroRef' provides global access to a non-const object; consider making the referenced data 'const'

#define DEFINE_NON_CONST_POINTER int *macroPtr = &nonConstInt;

DEFINE_NON_CONST_POINTER
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: variable 'macroPtr' is non-const and globally accessible, consider making it const
// CHECK-MESSAGES: :[[@LINE-2]]:1: warning: variable 'macroPtr' provides global access to a non-const object; consider making the pointed-to data 'const'
