// RUN: clang-tidy %s -checks='-*,llvm-namespace-comment' -- -std=c++17 | FileCheck %s

#pragma clang_tidy_analyze_as_header sysroot("/path/to/sysroot") header("<optional>") std("c++23")
#pragma clang_tidy_analyze_as_header sysroot("/path/to/sysroot") header("<vector>") std("c++23")

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
