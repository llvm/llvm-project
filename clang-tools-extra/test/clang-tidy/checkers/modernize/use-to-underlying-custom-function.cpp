// RUN: %check_clang_tidy -std=c++11-or-later %s modernize-use-to-underlying %t \
// RUN:   -- -config="{CheckOptions: { \
// RUN:        modernize-use-to-underlying.ReplacementFunction: 'llvm::to_underlying', \
// RUN:        modernize-use-to-underlying.ReplacementFunctionHeader: 'llvm/ADT/STLExtras.h'}}"

// With a user-provided replacement function the check runs before C++23 and
// uses the configured name and header.

// CHECK-FIXES: #include "llvm/ADT/STLExtras.h"

enum class E : int { A, B };

int convert(E e) {
  return static_cast<int>(e);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use 'llvm::to_underlying' to convert a scoped enumeration to its underlying type [modernize-use-to-underlying]
  // CHECK-FIXES: return llvm::to_underlying(e);
}
