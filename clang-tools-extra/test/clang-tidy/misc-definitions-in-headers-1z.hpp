// RUN: %check_clang_tidy %s misc-definitions-in-headers %t -- -- -std=c++1z

class CE {
  constexpr static int i = 5; // OK: inline variable definition.
};

inline int i = 5; // OK: inline variable definition.

int b = 1;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: variable 'b' defined in a header file; variable definitions in header files can lead to ODR violations [misc-definitions-in-headers]
