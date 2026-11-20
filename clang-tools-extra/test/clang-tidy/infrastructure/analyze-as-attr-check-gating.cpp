// The analyze_as attribute warnings are only given when a check that uses the
// attributes is enabled. Each case below says what happens with
// bugprone-unchecked-optional-access enabled and disabled.
//
// RUN: clang-tidy %s -checks='-*,bugprone-unchecked-optional-access' -- -std=c++17 -nostdinc++ -isystem %S/Inputs/analyze-as | FileCheck %s --check-prefix=ON --implicit-check-not=warning:
// A glob that includes the check counts as enabled.
// RUN: clang-tidy %s -checks='-*,bugprone-*' -- -std=c++17 -nostdinc++ -isystem %S/Inputs/analyze-as | FileCheck %s --check-prefix=ON --implicit-check-not=warning:
// RUN: clang-tidy %s -checks='-*,llvm-namespace-comment' -- -std=c++17 -nostdinc++ -isystem %S/Inputs/analyze-as | FileCheck %s --check-prefix=OFF --allow-empty --implicit-check-not=warning:
// OFF-NOT: warning:

// 1. Valid annotations: silent either way.
template <typename T>
class [[clang::analyze_as_class("std::optional")]] Valid {
public:
  [[clang::analyze_as_method("has_value")]] bool present() const;
  [[clang::analyze_as_method("value")]] T &get();
};

// 2. Misspelled method on a supported class.
//    Check enabled: reported.  Check disabled: silent.
template <typename T>
class [[clang::analyze_as_class("std::optional")]] MethodTypo {
public:
  [[clang::analyze_as_method("vaule")]] T &get();
  // ON: :[[@LINE-1]]:5: warning: 'analyze_as_method' argument "vaule" does not match anything in std::optional; the attribute is likely malformed [clang-tidy-config]
};

// 3. Unsupported (misspelled) class.
//    Check enabled: the class is reported, under the check's name.
//    Check disabled: silent.
//    Either way its methods are not validated, even when they are wrong too.
template <typename T>
class [[clang::analyze_as_class("std::optinal")]] ClassTypo {
  // ON: :[[@LINE-1]]:9: warning: 'analyze_as_class' argument "std::optinal" is not a supported class; supported: std::optional [bugprone-unchecked-optional-access]
public:
  [[clang::analyze_as_method("vaule")]] T &get();
};

// 4. Unsupported class with NOLINT for the check: suppressed like any other
//    warning of that check.
template <typename T>
class [[clang::analyze_as_class("std::optinal")]] ClassTypoNolint { // NOLINT(bugprone-unchecked-optional-access)
public:
  [[clang::analyze_as_method("vaule")]] T &get();
};

// 5. Method annotation on a class without analyze_as_class: no target class,
//    so nothing to validate against. Silent either way.
class NotAnnotated {
public:
  [[clang::analyze_as_method("vaule")]] int &get();
};
