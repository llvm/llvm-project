// RUN: %check_clang_tidy %s abseil-unchecked-statusor-access %t -- -header-filter='' -- -I %S/Inputs

#include "absl/status/statusor.h"
void unchecked_value_access(const absl::StatusOr<int>& sor) {
  sor.value();
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: unchecked access to 'absl::StatusOr' value [abseil-unchecked-statusor-access]
}

void unchecked_value_or_die_access(const absl::StatusOr<int>& sor) {
  sor.ValueOrDie();
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: unchecked access to 'absl::StatusOr' value [abseil-unchecked-statusor-access]
}

void unchecked_deref_operator_access(const absl::StatusOr<int>& sor) {
  *sor;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: unchecked access to 'absl::StatusOr' value [abseil-unchecked-statusor-access]
}

struct Foo {
  void foo() const {}
};

void unchecked_arrow_operator_access(const absl::StatusOr<Foo>& sor) {
  sor->foo();
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: unchecked access to 'absl::StatusOr' value [abseil-unchecked-statusor-access]
}

void f2(const absl::StatusOr<int>& sor) {
  if (sor.ok()) {
    sor.value();
  }
}

template <typename T>
void function_template_without_user(const absl::StatusOr<T>& sor) {
  sor.value(); // no-warning
}

template <typename T>
void function_template_with_user(const absl::StatusOr<T>& sor) {
  sor.value();
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: unchecked access to 'absl::StatusOr' value [abseil-unchecked-statusor-access]
}

void function_template_user(const absl::StatusOr<int>& sor) {
  // Instantiate the function_template_with_user function template so that it gets matched by the check.
  function_template_with_user(sor);
}

template<typename T>
void function_template_with_specialization(const absl::StatusOr<int>& sor) {
  sor.value(); // no-warning
}

template<>
void function_template_with_specialization<int>(const absl::StatusOr<int>& sor) {
  sor.value();
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: unchecked access to 'absl::StatusOr' value [abseil-unchecked-statusor-access]
}


template <typename T>
class ClassTemplateWithSpecializations {
  void f(const absl::StatusOr<int>& sor) {
    sor.value(); // no-warning
  }
};

template<typename T>
class ClassTemplateWithSpecializations<T*> {
  void f(const absl::StatusOr<int>& sor) {
    sor.value(); // no-warning
  }
};

template<>
class ClassTemplateWithSpecializations<int> {
  void f(const absl::StatusOr<int>& sor) {
    sor.value();
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: unchecked access to 'absl::StatusOr' value [abseil-unchecked-statusor-access]
  }
};

// The templates below are not instantiated and CFGs can not be properly built
// for them. They are here to make sure that the checker does not crash, but
// instead ignores non-instantiated templates.

template <typename T>
struct C1 {};

template <typename T>
struct C2 : public C1<T> {
  ~C2() {}
};

template <typename T, template <class> class B>
struct C3 : public B<T> {
  ~C3() {}
};

void multiple_unchecked_accesses(absl::StatusOr<int> sor1,
                                 absl::StatusOr<int> sor2) {
  for (int i = 0; i < 10; i++) {
    sor1.value();
    // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: unchecked access to 'absl::StatusOr' value [abseil-unchecked-statusor-access]
  }
  sor2.value();
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: unchecked access to 'absl::StatusOr' value [abseil-unchecked-statusor-access]
}

class C4 {
  explicit C4(absl::StatusOr<int> sor) : foo_(sor.value()) {
    // CHECK-MESSAGES: :[[@LINE-1]]:51: warning: unchecked access to 'absl::StatusOr' value [abseil-unchecked-statusor-access]
  }
  int foo_;
};

void lambda(const absl::StatusOr<int>& sor) {
  [&sor]() {
    sor.value();
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: unchecked access to 'absl::StatusOr' value [abseil-unchecked-statusor-access]
  }();

  [&]() {
    if (sor.ok()) {
      sor.value();
    }
  }();

  // Information from the surrounding context is not propagated through the
  // lambda.
  if (sor.ok()) {
    [&sor]() {
      sor.value();
      // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: unchecked access to 'absl::StatusOr' value [abseil-unchecked-statusor-access]
    }();
  }
}
