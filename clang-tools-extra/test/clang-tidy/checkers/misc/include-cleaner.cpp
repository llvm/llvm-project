// RUN: %check_clang_tidy %s misc-include-cleaner %t -- -- -I%S/Inputs -isystem%S/Inputs/system
#include "bar.h"
// CHECK-FIXES: #include "baz.h"
#include "foo.h"
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: included header foo.h is not used directly [misc-include-cleaner]
// CHECK-FIXES: {{^}}
// CHECK-FIXES: #include <string>
#include <vector.h>
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: included header vector.h is not used directly [misc-include-cleaner]
// CHECK-FIXES: {{^}}
int BarResult = bar();
int BazResult = baz();
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: no header providing "baz" is directly included [misc-include-cleaner]
std::string HelloString;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: no header providing "std::string" is directly included [misc-include-cleaner]
int FooBarResult = foobar();
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: no header providing "foobar" is directly included [misc-include-cleaner]
