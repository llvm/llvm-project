// The validator parses <optional> with this file's own compile flags, so the
// test supplies its own <optional> through -isystem; no path is hardcoded.
// RUN: clang-tidy %s -checks='-*,bugprone-unchecked-optional-access' -- -std=c++17 -nostdinc++ -isystem %S/Inputs/analyze-as | FileCheck %s --implicit-check-not=warning:
//
// The header AST is raised to C++17 even if this file is compiled as C++14.
// RUN: clang-tidy %s -checks='-*,bugprone-unchecked-optional-access' -- -std=c++14 -nostdinc++ -isystem %S/Inputs/analyze-as | FileCheck %s --implicit-check-not=warning:
//
// If <optional> cannot be found, nothing is validated (and nothing reported).
// RUN: clang-tidy %s -checks='-*,bugprone-unchecked-optional-access' -- -std=c++17 -nostdinc++ | count 0
//
// Which warnings depend on which checks are enabled is tested in
// analyze-as-attr-check-gating.cpp.

struct nothing_t {};

template <typename T>
class [[clang::analyze_as_class("std::optional")]] Good {
public:
  [[clang::analyze_as_method("has_value")]] bool present() const;
  [[clang::analyze_as_method("value")]] T &get();
  [[clang::analyze_as_method("emplace(Args&&...)")]] T &make();
  [[clang::analyze_as_method("optional(std::nullopt_t)")]] Good(nothing_t);
  [[clang::analyze_as_method("operator=(nullopt_t)")]] Good &operator=(nothing_t);
  [[clang::analyze_as_method("swap")]] void exchange(Good &);
  [[clang::analyze_as_method("reset")]] void clear();
};

// analyze_as_method is validated against the class its enclosing class is
// annotated as; methods of un-annotated classes have no target and are skipped.
class [[clang::analyze_as_class("std::optional")]] Bad {
public:
  [[clang::analyze_as_method("emplaceTYPO")]] void a();
  // CHECK: :[[@LINE-1]]:5: warning: 'analyze_as_method' argument "emplaceTYPO" does not match anything in std::optional; the attribute is likely malformed [clang-tidy-config]
  [[clang::analyze_as_method("swap(int, int, int)")]] void b();
  // CHECK: :[[@LINE-1]]:5: warning: 'analyze_as_method' argument "swap(int, int, int)" does not match anything in std::optional; the attribute is likely malformed [clang-tidy-config]
  [[clang::analyze_as_method("emplace(oops))")]] void c();
  // CHECK: :[[@LINE-1]]:5: warning: 'analyze_as_method' argument "emplace(oops))" does not match anything in std::optional; the attribute is likely malformed [clang-tidy-config]
  [[clang::analyze_as_method("value")]] void d(); // fine
};

class NotAnnotated {
public:
  [[clang::analyze_as_method("emplaceTYPO")]] void a(); // no target: skipped
};
