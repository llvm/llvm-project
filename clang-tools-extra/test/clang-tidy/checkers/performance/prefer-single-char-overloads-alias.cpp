// RUN: %check_clang_tidy %s performance-faster-string-find %t

#include <string>

void StringFind() {
  std::string Str;
  Str.find("a");
  // CHECK-MESSAGES: [[@LINE-1]]:12: warning: 'find' called with a string literal consisting of a single character; consider using the more efficient overload accepting a character [performance-faster-string-find]
  // CHECK-MESSAGES: [[@LINE-2]]:12: note: performance-faster-string-find is deprecated and will be removed in future release, consider using performance-prefer-single-char-overloads
  // CHECK-FIXES: Str.find('a');
}
