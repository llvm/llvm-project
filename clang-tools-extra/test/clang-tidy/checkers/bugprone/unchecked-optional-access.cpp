// RUN: %check_clang_tidy %s bugprone-unchecked-optional-access %t -- -- -I %S/Inputs/unchecked-optional-access

#include "absl/types/optional.h"
#include "folly/types/Optional.h"

void unchecked_value_access(const absl::optional<int> &opt) {
  opt.value();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: unchecked access to optional value [bugprone-unchecked-optional-access]
}

void unchecked_deref_operator_access(const absl::optional<int> &opt) {
  *opt;
  // CHECK-MESSAGES: :[[@LINE-1]]:4: warning: unchecked access to optional value
}

struct Foo {
  void foo() const {}
};

void unchecked_arrow_operator_access(const absl::optional<Foo> &opt) {
  opt->foo();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: unchecked access to optional value
}

void folly_check_value_then_reset(folly::Optional<int> opt) {
  if (opt) {
    opt.reset();
    opt.value();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: unchecked access to optional value
  }
}

void folly_value_after_swap(folly::Optional<int> opt1, folly::Optional<int> opt2) {
  if (opt1) {
    opt1.swap(opt2);
    opt1.value();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: unchecked access to optional value
  }
}

void checked_access(const absl::optional<int> &opt) {
  if (opt.has_value()) {
    opt.value();
  }
}

void folly_checked_access(const folly::Optional<int> &opt) {
  if (opt.hasValue()) {
    opt.value();
  }
}

template <typename T>
void function_template_without_user(const absl::optional<T> &opt) {
  opt.value(); // no-warning
}

template <typename T>
void function_template_with_user(const absl::optional<T> &opt) {
  opt.value();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: unchecked access to optional value
}

void function_template_user(const absl::optional<int> &opt) {
  // Instantiate the f3 function template so that it gets matched by the check.
  function_template_with_user(opt);
}

template <typename T>
void function_template_with_specialization(const absl::optional<int> &opt) {
  opt.value(); // no-warning
}

template <>
void function_template_with_specialization<int>(
    const absl::optional<int> &opt) {
  opt.value();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: unchecked access to optional value
}

template <typename T>
class ClassTemplateWithSpecializations {
  void f(const absl::optional<int> &opt) {
    opt.value(); // no-warning
  }
};

template <typename T>
class ClassTemplateWithSpecializations<T *> {
  void f(const absl::optional<int> &opt) {
    opt.value(); // no-warning
  }
};

template <>
class ClassTemplateWithSpecializations<int> {
  void f(const absl::optional<int> &opt) {
    opt.value();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: unchecked access to optional
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

void multiple_unchecked_accesses(absl::optional<int> opt1,
                                 absl::optional<int> opt2) {
  for (int i = 0; i < 10; i++) {
    opt1.value();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: unchecked access to optional
  }
  opt2.value();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: unchecked access to optional value
}

class C4 {
  explicit C4(absl::optional<int> opt) : foo_(opt.value()) {
    // CHECK-MESSAGES: :[[@LINE-1]]:47: warning: unchecked access to optional
  }
  int foo_;
};

// llvm#59705
namespace std
{
  template <typename T>
  constexpr T&& forward(T& type) noexcept {
    return static_cast<T&&>(type);
  }

  template <typename T>
  constexpr T&& forward(T&& type) noexcept {
    return static_cast<T&&>(type);
  }
}

void std_forward_copy(absl::optional<int> opt) {
  std::forward<absl::optional<int>>(opt).value();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: unchecked access to optional
}

void std_forward_copy_safe(absl::optional<int> opt) {
  if (!opt) return;

  std::forward<absl::optional<int>>(opt).value();
}

void std_forward_copy(absl::optional<int>& opt) {
  std::forward<absl::optional<int>>(opt).value();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: unchecked access to optional
}

void std_forward_lvalue_ref_safe(absl::optional<int>& opt) {
  if (!opt) return;

  std::forward<absl::optional<int>>(opt).value();
}

void std_forward_copy(absl::optional<int>&& opt) {
  std::forward<absl::optional<int>>(opt).value();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: unchecked access to optional
}

void std_forward_rvalue_ref_safe(absl::optional<int>&& opt) {
  if (!opt) return;

  std::forward<absl::optional<int>>(opt).value();
}
