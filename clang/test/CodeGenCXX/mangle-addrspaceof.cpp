// RUN: %clang_cc1 -std=c++20 -emit-llvm -o - %s | FileCheck %s
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
