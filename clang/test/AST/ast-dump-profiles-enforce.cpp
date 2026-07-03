// RUN: %clang_cc1 -std=c++23 -fprofiles -ast-dump %s | FileCheck %s

// A valid repetition of an already-recorded designator has no effect
// (P3589R2 [decl.attr.enforce]p3), so it must not re-append the designator to
// the attribute's argument arrays. The "already recorded" test must use the
// ungated enforcement list: a gated-off test:: profile (recorded but reported
// not-enforced without -fprofiles-test-profiles, which this run deliberately
// omits) is still a recorded designator.

[[profiles::enforce(test::type_cast)]];
// CHECK: EmptyDecl
// CHECK-NEXT: ProfilesEnforceAttr {{.*}} test::type_cast test::type_cast 0{{$}}

[[profiles::enforce(test::type_cast)]];
// CHECK: EmptyDecl
// CHECK-NEXT: ProfilesEnforceAttr {{.*}}>{{$}}

[[profiles::enforce(std::init, vendor(fortify: 3))]];
// CHECK: EmptyDecl
// CHECK-NEXT: ProfilesEnforceAttr {{.*}} std::init vendor std::init vendor(fortify : 3) 0 1 fortify 3 1{{$}}

[[profiles::enforce(std::init)]];
// CHECK: EmptyDecl
// CHECK-NEXT: ProfilesEnforceAttr {{.*}}>{{$}}
