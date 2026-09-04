// RUN: %check_clang_tidy -std=c++23-or-later %s modernize-use-to-underlying %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:       modernize-use-to-underlying.ReplacementFunction: 'std::to_underlying', \
// RUN:       modernize-use-to-underlying.ReplacementFunctionHeader: '<wrong>'}}"

// 'std::to_underlying' is declared in '<utility>', so configuring a different
// header is a misconfiguration and is reported. The check nevertheless proceeds
// and inserts the configured header.

// CHECK-MESSAGES: warning: 'std::to_underlying' is declared in '<utility>', but 'ReplacementFunctionHeader' is set to '<wrong>' [clang-tidy-config]
// CHECK-FIXES: #include <wrong>

enum class E : int { A, B };

int convert(E e) {
  return static_cast<int>(e);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use 'std::to_underlying' to convert a scoped enumeration to its underlying type [modernize-use-to-underlying]
  // CHECK-FIXES: return std::to_underlying(e);
}
