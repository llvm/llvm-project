// RUN: clang-tidy %s -checks='-*,clang-diagnostic-literal-conversion' --allow-no-checks -- -Wliteral-conversion | FileCheck %s

void f() {
  int i = 1.5;
  // CHECK: :[[@LINE-1]]:11: warning: implicit conversion from 'double' to 'int' changes value from 1.5 to 1 [clang-diagnostic-literal-conversion]
}
