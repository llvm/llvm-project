// RUN: %check_clang_tidy %s google-objc-function-naming %t

void printSomething() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function in global namespace named
// 'printSomething' must have an appropriate prefix followed by Pascal case as
// required by Google Objective-C style guide

void PrintSomething() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function in global namespace named
// 'PrintSomething' must have an appropriate prefix followed by Pascal case as
// required by Google Objective-C style guide

void ABCBad_Name() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function in global namespace named
// 'ABCBad_Name' must have an appropriate prefix followed by Pascal case as
// required by Google Objective-C style guide

namespace {

int foo() { return 0; }

}

namespace bar {

int convert() { return 0; }

}

class Baz {
public:
  int value() { return 0; }
};
