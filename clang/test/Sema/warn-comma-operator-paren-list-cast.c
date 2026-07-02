// RUN: %clang_cc1 -fsyntax-only -Wcomma -fno-caret-diagnostics %s 2>&1 | FileCheck %s

void comma_in_paren_list_cast(void) {
  int x;
  (void)(int)(x = 0, 1);
  // CHECK: :[[@LINE-1]]:20: warning: possible misuse of comma operator here
}
