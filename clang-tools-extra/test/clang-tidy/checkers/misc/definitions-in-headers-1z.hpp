// RUN: %check_clang_tidy -std=c++17-or-later %s misc-definitions-in-headers %t

class CE {
  constexpr static int i = 5; // OK: inline variable definition.
};

inline int i = 5; // OK: inline variable definition.

int b = 1;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: variable 'b' defined in a header file; variable definitions in header files can lead to ODR violations [misc-definitions-in-headers]

// OK: C++14 variable template.
template <class T>
constexpr T pi = T(3.1415926L);
