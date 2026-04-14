// RUN: %check_clang_tidy %s performance-faster-string-find %t
// RUN: %check_clang_tidy -check-suffix=CUSTOM %s performance-faster-string-find %t -- \
// RUN:   -config="{CheckOptions: \
// RUN:             {performance-faster-string-find.StringLikeClasses: \
// RUN:                '::llvm::StringRef;'}}"
// RUN: %check_clang_tidy -check-suffix=BOTH %s \
// RUN:   performance-faster-string-find,performance-prefer-single-char-overloads %t

#include <string>

namespace llvm {
struct StringRef {
  int find(const char *) const;
};
} // namespace llvm

void stringFind() {
  std::string Str;
  Str.find("a");
  // CHECK-MESSAGES: warning: 'performance-faster-string-find' check is deprecated and will be removed in a future release; consider using 'performance-prefer-single-char-overloads' instead [clang-tidy-config]
  // CHECK-MESSAGES: [[@LINE-2]]:12: warning: 'find' called with a string literal consisting of a single character; consider using the more efficient overload accepting a character [performance-faster-string-find]
  // CHECK-FIXES: Str.find('a');
  // CHECK-MESSAGES-BOTH: [[@LINE-4]]:12: warning: 'find' called with a string literal consisting of a single character; consider using the more efficient overload accepting a character
}

void customStringLikeClass() {
  llvm::StringRef Sr;
  Sr.find("x");
  // CHECK-MESSAGES-CUSTOM: warning: 'performance-faster-string-find' check is deprecated and will be removed in a future release; consider using 'performance-prefer-single-char-overloads' instead [clang-tidy-config]
  // CHECK-MESSAGES-CUSTOM: [[@LINE-2]]:11: warning: 'find' called with a string literal consisting of a single character; consider using the more efficient overload accepting a character [performance-faster-string-find]
  // CHECK-FIXES-CUSTOM: Sr.find('x');
}
