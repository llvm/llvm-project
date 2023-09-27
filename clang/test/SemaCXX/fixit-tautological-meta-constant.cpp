// RUN: %clang_cc1 -std=c++2b -Wno-unused-value -fdiagnostics-parseable-fixits -fsyntax-only %s 2>&1 | FileCheck %s
namespace std {
constexpr inline bool
  is_constant_evaluated() noexcept {
    if consteval { return true; } else { return false; }
  }
} // namespace std

constexpr void cexpr() {
  if constexpr (std::is_constant_evaluated()) {}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:6-[[@LINE-1]]:16}:""
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-2]]:6-[[@LINE-2]]:16}:""
  constexpr int a = std::is_constant_evaluated();
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:{{.*}}-[[@LINE-1]]:{{.*}}}:""

  if constexpr (const int ce = __builtin_is_constant_evaluated()) {}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:6-[[@LINE-1]]:16}:""
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-2]]:6-[[@LINE-2]]:16}:""
  constexpr int b = std::is_constant_evaluated();
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:{{.*}}-[[@LINE-1]]:{{.*}}}:""
}
