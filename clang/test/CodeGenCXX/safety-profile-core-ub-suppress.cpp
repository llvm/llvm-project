// RUN: %clang_cc1 -std=c++23 -fprofiles -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s

// std::core_ub (P4317): [[profiles::suppress(std::core_ub)]] is the in-source
// opt-out (SD-10 4.1). A whole-profile suppression on a function turns its
// guarded checks back off; a neighbouring unsuppressed function still traps.

[[profiles::enforce(std::core_ub)]];

// CHECK-LABEL: define {{.*}}@_Z9unguardedii
// CHECK-NOT: llvm.ubsantrap
// CHECK: {{^}}}
[[profiles::suppress(std::core_ub)]]
int unguarded(int a, int b) { return a / b; }

// CHECK-LABEL: define {{.*}}@_Z7guardedii
// CHECK: call void @llvm.ubsantrap(i8 3)
int guarded(int a, int b) { return a / b; }

// Suppression on an enclosing namespace covers the functions inside it.
namespace [[profiles::suppress(std::core_ub)]] ns {
// CHECK-LABEL: define {{.*}}@_ZN2ns6nestedEii
// CHECK-NOT: llvm.ubsantrap
// CHECK: {{^}}}
int nested(int a, int b) { return a / b; }
}
