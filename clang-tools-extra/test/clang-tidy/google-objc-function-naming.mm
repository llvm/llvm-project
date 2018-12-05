// RUN: %check_clang_tidy %s google-objc-function-naming %t

void printSomething() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function name 'printSomething' not
// using function naming conventions described by Google Objective-C style guide

void PrintSomething() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function name 'PrintSomething' not
// using function naming conventions described by Google Objective-C style guide

void ABCBad_Name() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function name 'ABCBad_Name' not
// using function naming conventions described by Google Objective-C style guide

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
