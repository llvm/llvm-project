// RUN: %check_clang_tidy -std=c++11 %s modernize-use-cpp-style-comments %t

static auto PI = 3.14159265; /* value of pi upto 8 decimal places */
// CHECK-MESSAGES: :[[@LINE-1]]:30: warning: use C++ style comments '//' instead of C style comments '/*...*/' [modernize-use-cpp-style-comments]

int a = /*some value */ 5;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use C++ style comments '//' instead of C style comments '/*...*/' [modernize-use-cpp-style-comments]