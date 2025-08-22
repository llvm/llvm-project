// RUN: %clang_cc1 -emit-llvm -triple x86_64-linux-gnu %s -o - -debug-info-kind=limited | FileCheck %s

template <typename LHS, typename RHS> constexpr bool is_same_v = false;
template <typename T> constexpr bool is_same_v<T, T> = true;

template constexpr bool is_same_v<int, int>;
static_assert(is_same_v<int, int>, "should get partial spec");

// Note that the template arguments for the instantiated variable use the
// parameter names from the primary template. The partial specialization might
// not have enough parameters.

// CHECK: distinct !DIGlobalVariable(name: "is_same_v", linkageName: "_Z9is_same_vIiiE", {{.*}} templateParams: ![[PARAMS:[0-9]+]])
// CHECK: ![[PARAMS]] = !{![[LHS:[0-9]+]], ![[RHS:[0-9]+]]}
// CHECK: ![[LHS]] = !DITemplateTypeParameter(name: "LHS", type: ![[INT:[0-9]+]])
// CHECK: ![[INT]] = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK: ![[RHS]] = !DITemplateTypeParameter(name: "RHS", type: ![[INT]])
