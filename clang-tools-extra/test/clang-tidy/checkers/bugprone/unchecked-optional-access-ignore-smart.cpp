// RUN: %check_clang_tidy %s bugprone-unchecked-optional-access %t -- \
// RUN:   -config="{CheckOptions: [ \
// RUN:     {key: bugprone-unchecked-optional-access.IgnoreSmartPointerDereference, value: true}]}" -- \
// RUN:   -I %S/Inputs/unchecked-optional-access

#include "absl/types/optional.h"

// Include some basic cases to ensure that IgnoreSmartPointerDereference doesn't
// disable everything. Then check the relevant smart-pointer cases.

void unchecked_deref_operator_access(const absl::optional<int> &opt) {
  *opt;
  // CHECK-MESSAGES: :[[@LINE-1]]:4: warning: unchecked access to optional value
}

void unchecked_value_access(const absl::optional<int> &opt) {
  opt.value();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: unchecked access to optional value [bugprone-unchecked-optional-access]
}

struct Foo {
  void foo() const {}
};

void unchecked_arrow_operator_access(const absl::optional<Foo> &opt) {
  opt->foo();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: unchecked access to optional value
}

template <typename T>
struct SmartPtr {
  T& operator*() &;
  T* operator->();
};

struct Bar {
  absl::optional<int> opt;
};


void unchecked_value_access_through_smart_ptr(SmartPtr<absl::optional<int>> s) {
  s->value();
  (*s).value();

}

void unchecked_value_access_through_smart_ptr_field(SmartPtr<Bar> s) {
  s->opt.value();
  (*s).opt.value();

}
