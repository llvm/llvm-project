// RUN: %check_clang_tidy %s performance-faster-string-find %t

#include <string>

void stringFind() {
  std::string Str;
  Str.find("a");
  // CHECK-MESSAGES: warning: 'performance-faster-string-find' check is deprecated and will be removed in a future release; consider using 'performance-prefer-single-char-overloads' instead [clang-tidy-config]
  // CHECK-MESSAGES: [[@LINE-2]]:12: warning: 'find' called with a string literal consisting of a single character; consider using the more efficient overload accepting a character [performance-faster-string-find]
  // CHECK-FIXES: Str.find('a');
}
