// RUN: %check_clang_tidy -check-suffixes=IGNORE -std=c++23-or-later %s \
// RUN:   modernize-use-to-underlying %t -- \
// RUN:   -config="{CheckOptions: {modernize-use-to-underlying.ImpreciseCasts: Ignore}}"
// RUN: %check_clang_tidy -check-suffixes=WARN -std=c++23-or-later %s \
// RUN:   modernize-use-to-underlying %t -- \
// RUN:   -config="{CheckOptions: {modernize-use-to-underlying.ImpreciseCasts: Warn}}"
// RUN: %check_clang_tidy -check-suffixes=PRESERVE -std=c++23-or-later %s \
// RUN:   modernize-use-to-underlying %t -- \
// RUN:   -config="{CheckOptions: {modernize-use-to-underlying.ImpreciseCasts: PreserveType}}"
// RUN: %check_clang_tidy -check-suffixes=UNDERLYING -std=c++23-or-later %s \
// RUN:   modernize-use-to-underlying %t -- \
// RUN:   -config="{CheckOptions: {modernize-use-to-underlying.ImpreciseCasts: UseUnderlyingType}}"

// CHECK-FIXES-WARN: #include <utility>
// CHECK-FIXES-PRESERVE: #include <utility>
// CHECK-FIXES-UNDERLYING: #include <utility>

namespace std {
template <typename T>
constexpr __underlying_type(T) to_underlying(T value) noexcept {
  return static_cast<__underlying_type(T)>(value);
}
} // namespace std

enum class E : int { A, B };

// A precise cast is always diagnosed and fully replaced, regardless of the
// ImpreciseCasts option.
int precise(E e) {
  return static_cast<int>(e);
  // CHECK-MESSAGES-IGNORE: :[[@LINE-1]]:10: warning: use 'std::to_underlying' to convert a scoped enumeration to its underlying type [modernize-use-to-underlying]
  // CHECK-MESSAGES-WARN: :[[@LINE-2]]:10: warning: use 'std::to_underlying' to convert a scoped enumeration to its underlying type [modernize-use-to-underlying]
  // CHECK-MESSAGES-PRESERVE: :[[@LINE-3]]:10: warning: use 'std::to_underlying' to convert a scoped enumeration to its underlying type [modernize-use-to-underlying]
  // CHECK-MESSAGES-UNDERLYING: :[[@LINE-4]]:10: warning: use 'std::to_underlying' to convert a scoped enumeration to its underlying type [modernize-use-to-underlying]
  // CHECK-FIXES-IGNORE: return std::to_underlying(e);
  // CHECK-FIXES-WARN: return std::to_underlying(e);
  // CHECK-FIXES-PRESERVE: return std::to_underlying(e);
  // CHECK-FIXES-UNDERLYING: return std::to_underlying(e);
}

// An imprecise cast:
//  - Ignore:            no diagnostic, no fix.
//  - Warn:              warn but offer no fix-it.
//  - PreserveType:      warn and wrap the operand, keeping the destination type.
//  - UseUnderlyingType: warn and replace the whole cast, changing the type.
long imprecise(E e) {
  return static_cast<long>(e);
  // CHECK-MESSAGES-WARN: :[[@LINE-1]]:10: warning: use 'std::to_underlying' to convert a scoped enumeration to its underlying type [modernize-use-to-underlying]
  // CHECK-MESSAGES-PRESERVE: :[[@LINE-2]]:10: warning: use 'std::to_underlying' to convert a scoped enumeration to its underlying type [modernize-use-to-underlying]
  // CHECK-MESSAGES-UNDERLYING: :[[@LINE-3]]:10: warning: use 'std::to_underlying' to convert a scoped enumeration to its underlying type [modernize-use-to-underlying]
  // CHECK-FIXES-IGNORE: return static_cast<long>(e);
  // CHECK-FIXES-WARN: return static_cast<long>(e);
  // CHECK-FIXES-PRESERVE: return static_cast<long>(std::to_underlying(e));
  // CHECK-FIXES-UNDERLYING: return std::to_underlying(e);
}
