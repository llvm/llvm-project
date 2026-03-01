// RUN: %check_clang_tidy -std=c++17-or-later %s bugprone-suspicious-stringview-data-usage %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     bugprone-suspicious-stringview-data-usage.StringViewTypes: '::custom::StrView', \
// RUN:     bugprone-suspicious-stringview-data-usage.AllowedCallees: '::safe_func;SafeClass' \
// RUN:   }}" -- -isystem %clang_tidy_headers
#include <string>

namespace custom {
struct StrView {
  const char *data();
  unsigned size();
};
} // namespace custom

void unsafe_func(const char *);
void safe_func(const char *);
struct SafeClass {
  SafeClass(const char *);
};
struct UnsafeClass {
  UnsafeClass(const char *);
};

void TestStringViewTypes(custom::StrView sv) {
  unsafe_func(sv.data());
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: result of a `data()` call may not be null terminated, provide size information to the callee to prevent potential issues [bugprone-suspicious-stringview-data-usage]
}

void TestAllowedCallees(custom::StrView sv) {
  safe_func(sv.data());
  SafeClass sc(sv.data());
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: result of a `data()` call may not be null terminated, provide size information to the callee to prevent potential issues [bugprone-suspicious-stringview-data-usage]
}

void TestNotAllowed(custom::StrView sv) {
  UnsafeClass uc(sv.data());
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: result of a `data()` call may not be null terminated, provide size information to the callee to prevent potential issues [bugprone-suspicious-stringview-data-usage]
}

void TestStdStringViewNotMatched(std::string_view sv) {
  unsafe_func(sv.data());
}
