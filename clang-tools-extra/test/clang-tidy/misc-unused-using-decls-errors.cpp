// RUN: %check_clang_tidy %s misc-unused-using-decls %t

namespace n {
class C;
}

using n::C;

void f() {
  for (C *p : unknown()) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:15: error: use of undeclared identifier 'unknown' [clang-diagnostic-error]
}
