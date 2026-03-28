// RUN: %check_clang_tidy %s performance-faster-string-find %t
// RUN: %check_clang_tidy -check-suffix=CUSTOM %s performance-faster-string-find %t -- \
// RUN:   -config="{CheckOptions: \
// RUN:             {performance-faster-string-find.StringLikeClasses: \
// RUN:                '::llvm::StringRef;'}}"

#include <string>

namespace llvm {
struct StringRef {
  int find(const char *) const;
};
} // namespace llvm

void StringFind() {
  std::string Str;
  Str.find("a");
  // CHECK-MESSAGES: [[@LINE-1]]:12: warning: 'find' called with a string literal consisting of a single character; consider using the more efficient overload accepting a character [performance-faster-string-find]
  // CHECK-MESSAGES: [[@LINE-2]]:12: note: performance-faster-string-find is deprecated and will be removed in future release, consider using performance-prefer-single-char-overloads
  // CHECK-FIXES: Str.find('a');

  llvm::StringRef sr;
  sr.find("x");
  // CHECK-MESSAGES-CUSTOM: [[@LINE-1]]:11: warning: 'find' called with a string literal consisting of a single character; consider using the more efficient overload accepting a character [performance-faster-string-find]
  // CHECK-MESSAGES-CUSTOM: [[@LINE-2]]:11: note: performance-faster-string-find is deprecated and will be removed in future release, consider using performance-prefer-single-char-overloads
  // CHECK-FIXES-CUSTOM: sr.find('x');
}
