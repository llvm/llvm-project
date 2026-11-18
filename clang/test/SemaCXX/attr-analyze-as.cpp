// RUN: %clang_cc1 -fsyntax-only -verify %s

// Sema keeps any non-empty analyze_as_class name: which names are supported is
// decided by the analysis tools (clang-tidy), not by the compiler. So a
// misspelled name is not diagnosed here.
template <typename T>
class [[clang::analyze_as_class("std::optional")]] Supported {
  [[clang::analyze_as_method("has_value")]] bool present() const;
};

template <typename T>
class [[clang::analyze_as_class("std::optinal")]] Misspelled {
  [[clang::analyze_as_method("vaule")]] T &get();
};

// Empty strings are rejected.
class [[clang::analyze_as_class("")]] EmptyClass {}; // expected-warning {{'clang::analyze_as_class' attribute argument not supported: }}

class C {
  [[clang::analyze_as_method("")]] void f(); // expected-warning {{'clang::analyze_as_method' attribute argument not supported: }}
};

// Subjects: classes for analyze_as_class, member functions for
// analyze_as_method.
[[clang::analyze_as_class("std::optional")]] int NotAClass; // expected-error {{'clang::analyze_as_class' attribute only applies to}}
[[clang::analyze_as_method("value")]] void notAMember(); // expected-error {{'clang::analyze_as_method' attribute only applies to}}

// The argument must be a string literal.
class [[clang::analyze_as_class(42)]] NotAString {}; // expected-error {{expected string literal as argument of 'analyze_as_class' attribute}}

// Only one of each per declaration.
class [[clang::analyze_as_class("std::optional"), clang::analyze_as_class("std::optional")]] Twice {}; // expected-error {{attribute 'clang::analyze_as_class' is already applied with different arguments}}
