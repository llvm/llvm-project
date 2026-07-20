// RUN: %clang_cc1 -std=c++23 -fprofiles -ast-dump %s | FileCheck %s

// The std::core_ub profile (P4317) is enforced through the same framework
// attribute as any other profile; enforcement is carried on the leading
// empty-declaration and recorded for CodeGen to read.

[[profiles::enforce(std::core_ub)]];
// CHECK: EmptyDecl
// CHECK-NEXT: ProfilesEnforceAttr {{.*}} std::core_ub std::core_ub 0{{$}}
