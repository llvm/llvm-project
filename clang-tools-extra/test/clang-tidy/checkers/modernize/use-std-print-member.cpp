// RUN: %check_clang_tidy %s modernize-use-std-print %t -- \
// RUN:   -config="{CheckOptions: \
// RUN:             { \
// RUN:               modernize-use-std-print.PrintfLikeFunctions: 'MyClass::printf', \
// RUN:               modernize-use-std-print.FprintfLikeFunctions: 'MyClass::fprintf', \
// RUN:               modernize-use-std-print.ReplacementPrintFunction: 'print', \
// RUN:               modernize-use-std-print.ReplacementPrintlnFunction: 'println', \
// RUN:             } \
// RUN:            }" \
// RUN:   -- -isystem %clang_tidy_headers

#include <cstdio>
#include <string.h>

struct MyStruct {};

struct MyClass
{
  template <typename... Args>
  void printf(const char *fmt, Args &&...);
  template <typename... Args>
  int fprintf(MyStruct *param1, const char *fmt, Args &&...);
};

void printf_simple(MyClass &myclass, MyClass *pmyclass) {
  myclass.printf("printf dot %d", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: myclass.print("printf dot {}", 42);

  pmyclass->printf("printf pointer %d", 43);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: pmyclass->print("printf pointer {}", 43);

  (*pmyclass).printf("printf deref pointer %d", 44);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: (*pmyclass).print("printf deref pointer {}", 44);
}

void printf_newline(MyClass &myclass, MyClass *pmyclass) {
  myclass.printf("printf dot newline %c\n", 'A');
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: myclass.println("printf dot newline {}", 'A');

  pmyclass->printf("printf pointer newline %c\n", 'B');
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: pmyclass->println("printf pointer newline {}", 'B');
}

void fprintf_simple(MyStruct *mystruct, MyClass &myclass, MyClass *pmyclass) {
  myclass.fprintf(mystruct, "fprintf dot %d", 142);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'print' instead of 'fprintf' [modernize-use-std-print]
  // CHECK-FIXES: myclass.print(mystruct, "fprintf dot {}", 142);

  pmyclass->fprintf(mystruct, "fprintf pointer %d", 143);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'print' instead of 'fprintf' [modernize-use-std-print]
  // CHECK-FIXES: pmyclass->print(mystruct, "fprintf pointer {}", 143);

  (*pmyclass).fprintf(mystruct, "fprintf deref pointer %d", 144);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'print' instead of 'fprintf' [modernize-use-std-print]
  // CHECK-FIXES: (*pmyclass).print(mystruct, "fprintf deref pointer {}", 144);
}

struct MyDerivedClass : public MyClass {};

void printf_derived(MyDerivedClass &derived)
{
  derived.printf("printf on derived class %d", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: derived.print("printf on derived class {}", 42);
}
