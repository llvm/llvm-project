// RUN: %clang_cc1 -fexperimental-new-constant-interpreter %s -Wno-constant-evaluated -verify
// RUN: %clang_cc1 -fexperimental-new-constant-interpreter %s -Wno-constant-evaluated -S -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -verify=ref %s -Wno-constant-evaluated
// RUN: %clang_cc1 -verify=ref %s -Wno-constant-evaluated %s -S -emit-llvm -o - | FileCheck %s

// expected-no-diagnostics
// ref-no-diagnostics

using size_t = decltype(sizeof(int));

namespace std {
inline constexpr bool is_constant_evaluated() noexcept {
  return __builtin_is_constant_evaluated();
}
} // namespace std

constexpr bool b = std::is_constant_evaluated();
static_assert(b, "");
static_assert(std::is_constant_evaluated() , "");


bool is_this_constant() {
  return __builtin_is_constant_evaluated(); // CHECK: ret i1 false
}

constexpr bool assume() {
  __builtin_assume(true);
  __builtin_assume(false);
  return true;
}
static_assert(assume(), "");
