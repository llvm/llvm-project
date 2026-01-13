// RUN: %check_clang_tidy %s bugprone-unchecked-optional-access %t -- \
// RUN:   -config="{CheckOptions:  \
// RUN:     {bugprone-unchecked-optional-access.IgnoreValueCalls: true}}" -- \
// RUN:   -I %S/Inputs/unchecked-optional-access

#include "absl/types/optional.h"

struct Foo {
  void foo() const {}
};

void unchecked_value_access(const absl::optional<int> &opt) {
  opt.value(); // no-warning
}

void unchecked_deref_operator_access(const absl::optional<int> &opt) {
  *opt;
  // CHECK-MESSAGES: :[[@LINE-1]]:4: warning: unchecked access to optional value
}

void unchecked_arrow_operator_access(const absl::optional<Foo> &opt) {
  opt->foo();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: unchecked access to optional value
}

