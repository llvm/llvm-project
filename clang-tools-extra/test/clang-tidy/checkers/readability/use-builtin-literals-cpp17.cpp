// RUN: %check_clang_tidy -std=c++17-or-later %s readability-use-builtin-literals %t

void warn_and_fix() {

  (char)u8'a';
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use built-in literal instead of explicit cast [readability-use-builtin-literals]
  // CHECK-FIXES: 'a';
  (wchar_t)u8'a';
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use built-in literal instead of explicit cast [readability-use-builtin-literals]
  // CHECK-FIXES: L'a';
}
