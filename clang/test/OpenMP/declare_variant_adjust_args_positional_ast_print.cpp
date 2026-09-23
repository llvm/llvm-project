// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 \
// RUN:   -x c++ -std=c++14 -fsyntax-only -verify %s

// expected-no-diagnostics

// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 \
// RUN:   -x c++ -std=c++14 -ast-print %s | FileCheck %s --check-prefix=PRINT

// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 \
// RUN:   -x c++ -std=c++14 -emit-pch -o %t %s

// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 \
// RUN:   -x c++ -std=c++14 -include-pch %t -ast-print %s \
// RUN:   | FileCheck %s --check-prefix=PRINT

// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 \
// RUN:   -x c++ -std=c++14 -ast-dump %s | FileCheck %s --check-prefix=DUMP

#ifndef HEADER
#define HEADER

void v_pos(int *A, int *B, int *C, int *D);
void v_range(int *A, int *B, int *C, int *D);
void v_variadic(int *A, int *B, int *C, int *D, ...);
void v_offset(int *A, int *B, int *C, ...);
void v_lb(int *A, int *B);
void v_ub(int *A, int *B);
void v_both(int *A, int *B);
void v_named(int *A, int *B);
void v_mixed(int *A, int *B);
void v_cond(int *A, int *B, int *C);
void v_addr(int &A, int &B);

// A list item may be a position: a constant integer expression.
// PRINT: #pragma omp declare variant(v_pos) match(construct={dispatch}) adjust_args(need_device_ptr:2,4)
// DUMP: FunctionDecl{{.*}}pos 'void (int *, int *, int *, int *)'
// DUMP: OMPDeclareVariantAttr
// DUMP: IntegerLiteral{{.*}}'int' 2
// DUMP-NEXT: IntegerLiteral{{.*}}'int' 4
#pragma omp declare variant(v_pos) match(construct={dispatch}) \
  adjust_args(need_device_ptr: 2, 4)
void pos(int *A, int *B, int *C, int *D);

// A list item may be a parameter range 'lb:ub'.
// PRINT: #pragma omp declare variant(v_range) match(construct={dispatch}) adjust_args(need_device_ptr:1:3)
// DUMP: FunctionDecl{{.*}}range 'void (int *, int *, int *, int *)'
// DUMP: OMPArgumentRangeExpr{{.*}}'void'
// DUMP-NEXT: IntegerLiteral{{.*}}'int' 1
// DUMP-NEXT: IntegerLiteral{{.*}}'int' 3
#pragma omp declare variant(v_range) match(construct={dispatch}) \
  adjust_args(need_device_ptr: 1:3)
void range(int *A, int *B, int *C, int *D);

// 'omp_num_args' is spelled only as a range bound, optionally with a logical
// offset. The offset is a constant expression, hence the ConstantExpr wrapper.
// PRINT: #pragma omp declare variant(v_variadic) match(construct={dispatch}) adjust_args(need_device_ptr:1:3,5,omp_num_args-1:omp_num_args)
// DUMP: FunctionDecl{{.*}}variadic 'void (int *, int *, int *, int *, ...)'
// DUMP: IntegerLiteral{{.*}}'int' 5
// DUMP-NEXT: OMPArgumentRangeExpr{{.*}}'void'
// DUMP-NEXT: OMPNumArgsExpr{{.*}}'int' '-'
// DUMP-NEXT: ConstantExpr{{.*}}'int'
// DUMP: IntegerLiteral{{.*}}'int' 1
// DUMP-NEXT: OMPNumArgsExpr{{.*}}'int'
// DUMP-NEXT: <<<NULL>>>
#pragma omp declare variant(v_variadic) match(construct={dispatch}) \
  adjust_args(need_device_ptr: 1:3, 5, omp_num_args-1:omp_num_args)
void variadic(int *A, int *B, int *C, int *D, ...);

// Both signs of the logical offset are accepted. An item that resolves outside
// the parameter list is ignored rather than diagnosed, per OpenMP 6.0 [9.6.2].
// PRINT: #pragma omp declare variant(v_offset) match(construct={dispatch}) adjust_args(nothing:omp_num_args-2:omp_num_args,omp_num_args+1:)
// DUMP: FunctionDecl{{.*}}offset 'void (int *, int *, int *, ...)'
// DUMP: OMPNumArgsExpr{{.*}}'int' '+'
#pragma omp declare variant(v_offset) match(construct={dispatch}) \
  adjust_args(nothing: omp_num_args-2:omp_num_args, omp_num_args+1:)
void offset(int *A, int *B, int *C, ...);

// OpenMP 6.0 [5.2.1]: an omitted 'lb' stands for 1. As the first list item
// it must still be separated from the adjust-op colon, or the two colons would
// lex as a single '::' and the printed clause would not parse back.
// PRINT: #pragma omp declare variant(v_lb) match(construct={dispatch}) adjust_args(need_device_ptr: :2)
// DUMP: FunctionDecl{{.*}}lb 'void (int *, int *)'
// DUMP: OMPArgumentRangeExpr{{.*}}'void'
// DUMP-NEXT: <<<NULL>>>
// DUMP-NEXT: IntegerLiteral{{.*}}'int' 2
#pragma omp declare variant(v_lb) match(construct={dispatch}) \
  adjust_args(need_device_ptr: :2)
void lb(int *A, int *B);

// An omitted 'ub' stands for 'omp_num_args'.
// PRINT: #pragma omp declare variant(v_ub) match(construct={dispatch}) adjust_args(need_device_ptr:1:)
// DUMP: FunctionDecl{{.*}}ub 'void (int *, int *)'
// DUMP: OMPArgumentRangeExpr{{.*}}'void'
// DUMP-NEXT: IntegerLiteral{{.*}}'int' 1
// DUMP-NEXT: <<<NULL>>>
#pragma omp declare variant(v_ub) match(construct={dispatch}) \
  adjust_args(need_device_ptr: 1:)
void ub(int *A, int *B);

// The two bounds are optional independently, so omitting both denotes
// 1:omp_num_args, that is every parameter.
// PRINT: #pragma omp declare variant(v_both) match(construct={dispatch}) adjust_args(need_device_ptr: :)
// DUMP: FunctionDecl{{.*}}both 'void (int *, int *)'
// DUMP: OMPArgumentRangeExpr{{.*}}'void'
// DUMP-NEXT: <<<NULL>>>
// DUMP-NEXT: <<<NULL>>>
#pragma omp declare variant(v_both) match(construct={dispatch}) \
  adjust_args(need_device_ptr: :)
void both(int *A, int *B);

// The pre-6.0 named form keeps its exact spelling and AST shape.
// PRINT: #pragma omp declare variant(v_named) match(construct={dispatch}) adjust_args(need_device_ptr:A,B)
// DUMP: FunctionDecl{{.*}}named 'void (int *, int *)'
// DUMP: DeclRefExpr{{.*}}'int *' lvalue ParmVar{{.*}}'A' 'int *'
// DUMP-NEXT: DeclRefExpr{{.*}}'int *' lvalue ParmVar{{.*}}'B' 'int *'
#pragma omp declare variant(v_named) match(construct={dispatch}) \
  adjust_args(need_device_ptr: A, B)
void named(int *A, int *B);

// Named items, positions and ranges may be mixed in one clause.
// PRINT: #pragma omp declare variant(v_mixed) match(construct={dispatch}) adjust_args(need_device_ptr:A,2:2)
// DUMP: FunctionDecl{{.*}}mixed 'void (int *, int *)'
// DUMP: DeclRefExpr{{.*}}'int *' lvalue ParmVar{{.*}}'A' 'int *'
// DUMP-NEXT: OMPArgumentRangeExpr{{.*}}'void'
#pragma omp declare variant(v_mixed) match(construct={dispatch}) \
  adjust_args(need_device_ptr: A, 2:2)
void mixed(int *A, int *B);

// A conditional operator keeps its own colon: the item is one position, not a
// range, so a bound is never split at a '?:' colon.
// PRINT: #pragma omp declare variant(v_cond) match(construct={dispatch}) adjust_args(need_device_ptr:1 ? 2 : 3)
// DUMP: FunctionDecl{{.*}}cond 'void (int *, int *, int *)'
// DUMP: ConditionalOperator{{.*}}'int'
#pragma omp declare variant(v_cond) match(construct={dispatch}) \
  adjust_args(need_device_ptr: 1 ? 2 : 3)
void cond(int *A, int *B, int *C);

// A range does not name a parameter, so the 'need_device_addr' reference-type
// restriction of OpenMP 6.0 [9.6.2] does not apply to it.
// PRINT: #pragma omp declare variant(v_addr) match(construct={dispatch}) adjust_args(need_device_addr:1:2)
#pragma omp declare variant(v_addr) match(construct={dispatch}) \
  adjust_args(need_device_addr: 1:2)
void addr(int &A, int &B);

// Dependent bounds are accepted in the template pattern and rechecked on
// instantiation.
template <int N>
void tmpl_v(int *A, int *B, int *C, ...);

template <int N>
void tmpl(int *A, int *B, int *C, ...);

// The instantiation is dumped before the pattern it came from, so its checks
// come first. Substituting N=2 makes the logical offset a constant expression.
// DUMP: FunctionDecl{{.*}}tmpl 'void (int *, int *, int *, ...)' explicit_instantiation_definition
// DUMP: OMPArgumentRangeExpr{{.*}}'void'
// DUMP-NEXT: SubstNonTypeTemplateParmExpr{{.*}}'int'
// DUMP: BinaryOperator{{.*}}'int' '+'
// DUMP: OMPNumArgsExpr{{.*}}'int' '-'
// DUMP-NEXT: ConstantExpr{{.*}}'int'
// DUMP-NEXT: value: Int 2
//
// In the pattern the offset stays dependent, so it is not wrapped.
// DUMP: OMPNumArgsExpr{{.*}}'int' '-'
// DUMP-NEXT: DeclRefExpr{{.*}}'int' NonTypeTemplateParm{{.*}}'N' 'int'
//
// PRINT: #pragma omp declare variant(tmpl_v<N>) match(construct={dispatch}) adjust_args(need_device_ptr:N:N + 1,omp_num_args-N:omp_num_args)
#pragma omp declare variant(tmpl_v<N>) match(construct={dispatch}) \
  adjust_args(need_device_ptr: N:N + 1, omp_num_args-N:omp_num_args)
template <int N>
void tmpl(int *A, int *B, int *C, ...) {}

template void tmpl<2>(int *, int *, int *, ...);

#endif // HEADER
