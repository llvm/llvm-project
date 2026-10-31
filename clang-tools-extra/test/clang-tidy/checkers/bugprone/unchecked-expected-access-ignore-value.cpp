// RUN: %check_clang_tidy -std=c++23-or-later %s bugprone-unchecked-expected-access %t -- \
// RUN:   -config="{CheckOptions:  \
// RUN:     {bugprone-unchecked-expected-access.IgnoreValueCalls: true}}" -- \
// RUN:   -I %S/Inputs/unchecked-expected-access

#include "std/types/expected.h"

struct Foo {
  void foo() const {}
};

void unchecked_value_access(const std::expected<int, int> &e) {
  e.value(); // no-warning
}

void unchecked_deref_operator_access(const std::expected<int, int> &e) {
  *e;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: unchecked access to 'std::expected' value [bugprone-unchecked-expected-access]
}

void unchecked_arrow_operator_access(const std::expected<Foo, int> &e) {
  e->foo();
  // CHECK-MESSAGES: :[[@LINE-1]]:4: warning: unchecked access to 'std::expected' value [bugprone-unchecked-expected-access]
}

void unchecked_error_access(const std::expected<int, int> &e) {
  e.error();
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: unchecked access to 'std::expected' error [bugprone-unchecked-expected-access]
}
