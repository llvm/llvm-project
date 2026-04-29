// RUN: %check_clang_tidy -check-suffixes=ALL -std=c++11-or-later %s bugprone-exception-escape %t -- \
// RUN:     -config='{"CheckOptions": { \
// RUN:       "bugprone-exception-escape.TreatFunctionsWithoutSpecificationAsThrowing": "All" \
// RUN:     }}' -- -fexceptions
// RUN: %check_clang_tidy -check-suffixes=UNDEFINED -std=c++11-or-later %s bugprone-exception-escape %t -- \
// RUN:     -config='{"CheckOptions": { \
// RUN:       "bugprone-exception-escape.TreatFunctionsWithoutSpecificationAsThrowing": "OnlyUndefined" \
// RUN:     }}' -- -fexceptions
// RUN: %check_clang_tidy -check-suffixes=NONE -std=c++11-or-later %s bugprone-exception-escape %t -- \
// RUN:     -config='{"CheckOptions": { \
// RUN:       "bugprone-exception-escape.TreatFunctionsWithoutSpecificationAsThrowing": "None" \
// RUN:     }}' -- -fexceptions

#include <string>

void unannotated_no_throw_body() {}

void calls_unannotated() noexcept {
  // CHECK-MESSAGES-ALL: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'calls_unannotated' which should not throw exceptions
  // CHECK-MESSAGES-ALL: :[[@LINE-4]]:6: note: frame #0: an exception of unknown type may be thrown in function 'unannotated_no_throw_body' here
  // CHECK-MESSAGES-ALL: :[[@LINE+3]]:3: note: frame #1: function 'calls_unannotated' calls function 'unannotated_no_throw_body' here
  // CHECK-MESSAGES-UNDEFINED-NOT: warning:
  // CHECK-MESSAGES-NONE-NOT: warning:
  unannotated_no_throw_body();
}

void extern_declared();

void calls_unknown() noexcept {
  // CHECK-MESSAGES-ALL: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'calls_unknown' which should not throw exceptions
  // CHECK-MESSAGES-ALL: :[[@LINE-4]]:6: note: frame #0: an exception of unknown type may be thrown in function 'extern_declared' here
  // CHECK-MESSAGES-ALL: :[[@LINE+5]]:3: note: frame #1: function 'calls_unknown' calls function 'extern_declared' here
  // CHECK-MESSAGES-UNDEFINED: :[[@LINE-4]]:6: warning: an exception may be thrown in function 'calls_unknown' which should not throw exceptions
  // CHECK-MESSAGES-UNDEFINED: :[[@LINE-7]]:6: note: frame #0: an exception of unknown type may be thrown in function 'extern_declared' here
  // CHECK-MESSAGES-UNDEFINED: :[[@LINE+2]]:3: note: frame #1: function 'calls_unknown' calls function 'extern_declared' here
  // CHECK-MESSAGES-NONE-NOT: warning:
  extern_declared();
}

void calls_unknown_caught() noexcept {
  // CHECK-MESSAGES-ALL-NOT: warning:
  // CHECK-MESSAGES-UNDEFINED-NOT: warning:
  // CHECK-MESSAGES-NONE-NOT: warning:
  try {
    extern_declared();
  } catch(...) {
  }
}

void definitely_nothrow() noexcept {}

void calls_nothrow() noexcept {
  // CHECK-MESSAGES-ALL-NOT: warning:
  // CHECK-MESSAGES-UNDEFINED-NOT: warning:
  // CHECK-MESSAGES-NONE-NOT: warning:
  definitely_nothrow();
}

void nothrow_nobody() throw();

void call() noexcept {
  nothrow_nobody();
}

struct Member {
  Member() noexcept {}
  Member(const Member &) noexcept {}
  Member &operator=(const Member &) noexcept { return *this; }
  ~Member() noexcept {}
};

struct S {
  Member m;
};

template <typename T>
struct TmplNoexcept {
  void method() noexcept(noexcept(T())) {}
};

void instantiate_tmpl_noexcept() {
  TmplNoexcept<int> t;
}

void explicit_throw() { throw 1; }
void calls_explicit_throw() noexcept {
  // CHECK-MESSAGES-ALL: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'calls_explicit_throw' which should not throw exceptions
  // CHECK-MESSAGES-ALL: :[[@LINE-3]]:25: note: frame #0: unhandled exception of type 'int' may be thrown in function 'explicit_throw' here
  // CHECK-MESSAGES-ALL: :[[@LINE+7]]:3: note: frame #1: function 'calls_explicit_throw' calls function 'explicit_throw' here
  // CHECK-MESSAGES-UNDEFINED: :[[@LINE-4]]:6: warning: an exception may be thrown in function 'calls_explicit_throw' which should not throw exceptions
  // CHECK-MESSAGES-UNDEFINED: :[[@LINE-6]]:25: note: frame #0: unhandled exception of type 'int' may be thrown in function 'explicit_throw' here
  // CHECK-MESSAGES-UNDEFINED: :[[@LINE+4]]:3: note: frame #1: function 'calls_explicit_throw' calls function 'explicit_throw' here
  // CHECK-MESSAGES-NONE: :[[@LINE-7]]:6: warning: an exception may be thrown in function 'calls_explicit_throw' which should not throw exceptions
  // CHECK-MESSAGES-NONE: :[[@LINE-9]]:25: note: frame #0: unhandled exception of type 'int' may be thrown in function 'explicit_throw' here
  // CHECK-MESSAGES-NONE: :[[@LINE+1]]:3: note: frame #1: function 'calls_explicit_throw' calls function 'explicit_throw' here
  explicit_throw();
}

struct ImplicitDtor {
  ImplicitDtor() = default;
};

struct DefaultedDtor {
  DefaultedDtor() = default;
  ~DefaultedDtor() = default;
};

struct WithString {
  WithString(const ImplicitDtor &Implicit, const DefaultedDtor &Defaulted,
             const std::string &Text)
      : Implicit(Implicit), Defaulted(Defaulted), Text(Text) {}

  ImplicitDtor Implicit;
  DefaultedDtor Defaulted;
  std::string Text;
};

void constructs_with_string() {
  ImplicitDtor Implicit;
  DefaultedDtor Defaulted;
  WithString Value(Implicit, Defaulted, "");
}
