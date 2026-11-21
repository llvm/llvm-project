// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -std=c++20 -emit-llvm -o - %s | FileCheck %s --check-prefix=MS
// RUN: %clang_cc1 -std=c++20 -ast-print %s | FileCheck %s --check-prefix=PRINT

int global;
constexpr int direct_entity = __addrspaceof(global);
constexpr int parenthesized_expression = __addrspaceof((global));

// PRINT: constexpr int direct_entity = __addrspaceof(global);
// PRINT: constexpr int parenthesized_expression = __addrspaceof((global));

template <class T> void type_operand(decltype(__addrspaceof(T))) {}
template void type_operand<int>(int);

template <class T>
void expression_operand(T &value, decltype(__addrspaceof(value))) {}
template void expression_operand<int>(int &, int);

// CHECK-DAG: define weak_odr void @_Z12type_operandIiEvDTu13__addrspaceofT_EE(
// The boolean template argument records the entity form because ordinary
// expression mangling does not preserve parentheses.
// CHECK-DAG: define weak_odr void @_Z18expression_operandIiEvRT_DTu13__addrspaceofLb1EXfL0p_EEE(
// MS-DAG: define weak_odr dso_local void @"??$type_operand@H@@YAXH@Z"(
// MS-DAG: define weak_odr dso_local void @"??$expression_operand@H@@YAXAEAHH@Z"(

using AS1 = int __attribute__((address_space(1)));
template <int N> int value_operand() { return N; }
template int value_operand<__addrspaceof(AS1)>();

// MS: define weak_odr dso_local noundef i32 @"??$value_operand@$0BAAAAAB@@@YAHXZ"(
// MS: ret i32 16777217
