// RUN: %clang_cc1 -std=c++20 -fexperimental-cxx-profiles -ast-dump %s \
// RUN:   | FileCheck %s

[[profiles::enforce(std::casting)]];
// CHECK: EmptyDecl
// CHECK-NEXT: ProfilesEnforceAttr {{.*}} "std::casting"

[[profiles::exempt(std::casting, angle_header: "legacy.h",
                   justification: "C interface")]];
// CHECK: EmptyDecl
// CHECK-NEXT: ProfilesExemptAttr {{.*}} "std::casting" "legacy.h"

void f(int *p) {
  [[profiles::suppress(std::casting, rule: reinterpret_cast,
                       justification: "testing")]]
  (void)reinterpret_cast<char *>(p);
}
// CHECK: AttributedStmt
// CHECK-NEXT: ProfilesSuppressAttr {{.*}} "std::casting" "reinterpret_cast" "testing"
