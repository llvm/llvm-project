// FileCheck lines automatically generated using make-ast-dump-check-v2.py

// RUN: %clang_cc1 -ast-dump -fbounds-safety -fbounds-safety-bringup-missing-checks=ended_by_lower_bound -verify %s 2> /dev/null | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -fbounds-safety-bringup-missing-checks=ended_by_lower_bound -verify -x objective-c -fexperimental-bounds-safety-objc %s 2> /dev/null | FileCheck %s
#include <ptrcheck.h>

// expected-no-diagnostics

// CHECK-LABEL:|-FunctionDecl {{.+}} <{{.+}}:12:1, col:{{.+}}> col:{{.+}} used ended_by 'void (const char *__single __ended_by(end), const char *__single /* __started_by(start) */ )'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used start 'const char *__single __ended_by(end)':'const char *__single'
// CHECK-NEXT: | `-ParmVarDecl {{.+}} used end 'const char *__single /* __started_by(start) */ ':'const char *__single'
void ended_by(const char *__ended_by(end) start, const char *end);

// CHECK-LABEL:|-FunctionDecl {{.+}} pass_const_size_arr_in_bounds 'void (void)'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   |-DeclStmt {{.+}}
// CHECK-NEXT: |   | `-VarDecl {{.+}} used local 'char[10]'
// CHECK-NEXT: |   `-MaterializeSequenceExpr {{.+}} 'void' <Unbind>
// CHECK-NEXT: |     |-MaterializeSequenceExpr {{.+}} 'void' <Bind>
// CHECK-NEXT: |     | |-BoundsCheckExpr {{.+}} 'void' '&local[10] <= __builtin_get_pointer_upper_bound(local) && local <= &local[10] && __builtin_get_pointer_lower_bound(local) <= local'
// CHECK-NEXT: |     | | |-CallExpr {{.+}} 'void'
// CHECK-NEXT: |     | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(const char *__single __ended_by(end), const char *__single /* __started_by(start) */ )' <FunctionToPointerDecay>
// CHECK-NEXT: |     | | | | `-DeclRefExpr {{.+}} 'void (const char *__single __ended_by(end), const char *__single /* __started_by(start) */ )' Function {{.+}} 'ended_by' 'void (const char *__single __ended_by(end), const char *__single /* __started_by(start) */ )'
// CHECK-NEXT: |     | | | |-ImplicitCastExpr {{.+}} 'const char *__single __ended_by(end)':'const char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | | | | `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | | | |   `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | | | |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: |     | | | |       `-DeclRefExpr {{.+}} 'char[10]' lvalue Var {{.+}} 'local' 'char[10]'
// CHECK-NEXT: |     | | | `-ImplicitCastExpr {{.+}} 'const char *__single /* __started_by(start) */ ':'const char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | | |   `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | | |     `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | | |       `-UnaryOperator {{.+}} 'char *__bidi_indexable' prefix '&' cannot overflow
// CHECK-NEXT: |     | | |         `-ArraySubscriptExpr {{.+}} 'char' lvalue
// CHECK-NEXT: |     | | |           |-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: |     | | |           | `-DeclRefExpr {{.+}} 'char[10]' lvalue Var {{.+}} 'local' 'char[10]'
// CHECK-NEXT: |     | | |           `-IntegerLiteral {{.+}} 'int' 10
// CHECK-NEXT: |     | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |     | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |     | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |     | |   | | |-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | |   | | | `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | |   | | |   `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | |   | | |     `-UnaryOperator {{.+}} 'char *__bidi_indexable' prefix '&' cannot overflow
// CHECK-NEXT: |     | |   | | |       `-ArraySubscriptExpr {{.+}} 'char' lvalue
// CHECK-NEXT: |     | |   | | |         |-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: |     | |   | | |         | `-DeclRefExpr {{.+}} 'char[10]' lvalue Var {{.+}} 'local' 'char[10]'
// CHECK-NEXT: |     | |   | | |         `-IntegerLiteral {{.+}} 'int' 10
// CHECK-NEXT: |     | |   | | `-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | |   | |   `-GetBoundExpr {{.+}} 'const char *__bidi_indexable' upper
// CHECK-NEXT: |     | |   | |     `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | |   | |       `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | |   | |         `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: |     | |   | |           `-DeclRefExpr {{.+}} 'char[10]' lvalue Var {{.+}} 'local' 'char[10]'
// CHECK-NEXT: |     | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |     | |   |   |-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | |   |   | `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | |   |   |   `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | |   |   |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: |     | |   |   |       `-DeclRefExpr {{.+}} 'char[10]' lvalue Var {{.+}} 'local' 'char[10]'
// CHECK-NEXT: |     | |   |   `-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | |   |     `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | |   |       `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | |   |         `-UnaryOperator {{.+}} 'char *__bidi_indexable' prefix '&' cannot overflow
// CHECK-NEXT: |     | |   |           `-ArraySubscriptExpr {{.+}} 'char' lvalue
// CHECK-NEXT: |     | |   |             |-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: |     | |   |             | `-DeclRefExpr {{.+}} 'char[10]' lvalue Var {{.+}} 'local' 'char[10]'
// CHECK-NEXT: |     | |   |             `-IntegerLiteral {{.+}} 'int' 10
// CHECK-NEXT: |     | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |     | |     |-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | |     | `-GetBoundExpr {{.+}} 'const char *__bidi_indexable' lower
// CHECK-NEXT: |     | |     |   `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | |     |     `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | |     |       `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: |     | |     |         `-DeclRefExpr {{.+}} 'char[10]' lvalue Var {{.+}} 'local' 'char[10]'
// CHECK-NEXT: |     | |     `-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | |       `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | |         `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | |           `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: |     | |             `-DeclRefExpr {{.+}} 'char[10]' lvalue Var {{.+}} 'local' 'char[10]'
// CHECK-NEXT: |     | |-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | | `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: |     | |     `-DeclRefExpr {{.+}} 'char[10]' lvalue Var {{.+}} 'local' 'char[10]'
// CHECK-NEXT: |     | `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     |   `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     |     `-UnaryOperator {{.+}} 'char *__bidi_indexable' prefix '&' cannot overflow
// CHECK-NEXT: |     |       `-ArraySubscriptExpr {{.+}} 'char' lvalue
// CHECK-NEXT: |     |         |-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: |     |         | `-DeclRefExpr {{.+}} 'char[10]' lvalue Var {{.+}} 'local' 'char[10]'
// CHECK-NEXT: |     |         `-IntegerLiteral {{.+}} 'int' 10
// CHECK-NEXT: |     |-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: |     |     `-DeclRefExpr {{.+}} 'char[10]' lvalue Var {{.+}} 'local' 'char[10]'
// CHECK-NEXT: |     `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |       `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |         `-UnaryOperator {{.+}} 'char *__bidi_indexable' prefix '&' cannot overflow
// CHECK-NEXT: |           `-ArraySubscriptExpr {{.+}} 'char' lvalue
// CHECK-NEXT: |             |-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: |             | `-DeclRefExpr {{.+}} 'char[10]' lvalue Var {{.+}} 'local' 'char[10]'
// CHECK-NEXT: |             `-IntegerLiteral {{.+}} 'int' 10
void pass_const_size_arr_in_bounds(void) {
  char local[10];
  ended_by(local, &local[10]);
}

// CHECK-LABEL:|-FunctionDecl {{.+}} pass_const_size_arr_start_oob 'void (void)'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   |-DeclStmt {{.+}}
// CHECK-NEXT: |   | `-VarDecl {{.+}} used local 'char[10]'
// CHECK-NEXT: |   `-MaterializeSequenceExpr {{.+}} 'void' <Unbind>
// CHECK-NEXT: |     |-MaterializeSequenceExpr {{.+}} 'void' <Bind>
// CHECK-NEXT: |     | |-BoundsCheckExpr {{.+}} 'void' '&local[10] <= __builtin_get_pointer_upper_bound(local - 2) && local - 2 <= &local[10] && __builtin_get_pointer_lower_bound(local - 2) <= local - 2'
// CHECK-NEXT: |     | | |-CallExpr {{.+}} 'void'
// CHECK-NEXT: |     | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(const char *__single __ended_by(end), const char *__single /* __started_by(start) */ )' <FunctionToPointerDecay>
// CHECK-NEXT: |     | | | | `-DeclRefExpr {{.+}} 'void (const char *__single __ended_by(end), const char *__single /* __started_by(start) */ )' Function {{.+}} 'ended_by' 'void (const char *__single __ended_by(end), const char *__single /* __started_by(start) */ )'
// CHECK-NEXT: |     | | | |-ImplicitCastExpr {{.+}} 'const char *__single __ended_by(end)':'const char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | | | | `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | | | |   `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | | | |     `-BinaryOperator {{.+}} 'char *__bidi_indexable' '-'
// CHECK-NEXT: |     | | | |       |-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: |     | | | |       | `-DeclRefExpr {{.+}} 'char[10]' lvalue Var {{.+}} 'local' 'char[10]'
// CHECK-NEXT: |     | | | |       `-IntegerLiteral {{.+}} 'int' 2
// CHECK-NEXT: |     | | | `-ImplicitCastExpr {{.+}} 'const char *__single /* __started_by(start) */ ':'const char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | | |   `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | | |     `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | | |       `-UnaryOperator {{.+}} 'char *__bidi_indexable' prefix '&' cannot overflow
// CHECK-NEXT: |     | | |         `-ArraySubscriptExpr {{.+}} 'char' lvalue
// CHECK-NEXT: |     | | |           |-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: |     | | |           | `-DeclRefExpr {{.+}} 'char[10]' lvalue Var {{.+}} 'local' 'char[10]'
// CHECK-NEXT: |     | | |           `-IntegerLiteral {{.+}} 'int' 10
// CHECK-NEXT: |     | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |     | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |     | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |     | |   | | |-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | |   | | | `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | |   | | |   `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | |   | | |     `-UnaryOperator {{.+}} 'char *__bidi_indexable' prefix '&' cannot overflow
// CHECK-NEXT: |     | |   | | |       `-ArraySubscriptExpr {{.+}} 'char' lvalue
// CHECK-NEXT: |     | |   | | |         |-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: |     | |   | | |         | `-DeclRefExpr {{.+}} 'char[10]' lvalue Var {{.+}} 'local' 'char[10]'
// CHECK-NEXT: |     | |   | | |         `-IntegerLiteral {{.+}} 'int' 10
// CHECK-NEXT: |     | |   | | `-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | |   | |   `-GetBoundExpr {{.+}} 'const char *__bidi_indexable' upper
// CHECK-NEXT: |     | |   | |     `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | |   | |       `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | |   | |         `-BinaryOperator {{.+}} 'char *__bidi_indexable' '-'
// CHECK-NEXT: |     | |   | |           |-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: |     | |   | |           | `-DeclRefExpr {{.+}} 'char[10]' lvalue Var {{.+}} 'local' 'char[10]'
// CHECK-NEXT: |     | |   | |           `-IntegerLiteral {{.+}} 'int' 2
// CHECK-NEXT: |     | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |     | |   |   |-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | |   |   | `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | |   |   |   `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | |   |   |     `-BinaryOperator {{.+}} 'char *__bidi_indexable' '-'
// CHECK-NEXT: |     | |   |   |       |-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: |     | |   |   |       | `-DeclRefExpr {{.+}} 'char[10]' lvalue Var {{.+}} 'local' 'char[10]'
// CHECK-NEXT: |     | |   |   |       `-IntegerLiteral {{.+}} 'int' 2
// CHECK-NEXT: |     | |   |   `-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | |   |     `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | |   |       `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | |   |         `-UnaryOperator {{.+}} 'char *__bidi_indexable' prefix '&' cannot overflow
// CHECK-NEXT: |     | |   |           `-ArraySubscriptExpr {{.+}} 'char' lvalue
// CHECK-NEXT: |     | |   |             |-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: |     | |   |             | `-DeclRefExpr {{.+}} 'char[10]' lvalue Var {{.+}} 'local' 'char[10]'
// CHECK-NEXT: |     | |   |             `-IntegerLiteral {{.+}} 'int' 10
// CHECK-NEXT: |     | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |     | |     |-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | |     | `-GetBoundExpr {{.+}} 'const char *__bidi_indexable' lower
// CHECK-NEXT: |     | |     |   `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | |     |     `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | |     |       `-BinaryOperator {{.+}} 'char *__bidi_indexable' '-'
// CHECK-NEXT: |     | |     |         |-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: |     | |     |         | `-DeclRefExpr {{.+}} 'char[10]' lvalue Var {{.+}} 'local' 'char[10]'
// CHECK-NEXT: |     | |     |         `-IntegerLiteral {{.+}} 'int' 2
// CHECK-NEXT: |     | |     `-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | |       `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | |         `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | |           `-BinaryOperator {{.+}} 'char *__bidi_indexable' '-'
// CHECK-NEXT: |     | |             |-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: |     | |             | `-DeclRefExpr {{.+}} 'char[10]' lvalue Var {{.+}} 'local' 'char[10]'
// CHECK-NEXT: |     | |             `-IntegerLiteral {{.+}} 'int' 2
// CHECK-NEXT: |     | |-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | | `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | |   `-BinaryOperator {{.+}} 'char *__bidi_indexable' '-'
// CHECK-NEXT: |     | |     |-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: |     | |     | `-DeclRefExpr {{.+}} 'char[10]' lvalue Var {{.+}} 'local' 'char[10]'
// CHECK-NEXT: |     | |     `-IntegerLiteral {{.+}} 'int' 2
// CHECK-NEXT: |     | `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     |   `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     |     `-UnaryOperator {{.+}} 'char *__bidi_indexable' prefix '&' cannot overflow
// CHECK-NEXT: |     |       `-ArraySubscriptExpr {{.+}} 'char' lvalue
// CHECK-NEXT: |     |         |-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: |     |         | `-DeclRefExpr {{.+}} 'char[10]' lvalue Var {{.+}} 'local' 'char[10]'
// CHECK-NEXT: |     |         `-IntegerLiteral {{.+}} 'int' 10
// CHECK-NEXT: |     |-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     |   `-BinaryOperator {{.+}} 'char *__bidi_indexable' '-'
// CHECK-NEXT: |     |     |-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: |     |     | `-DeclRefExpr {{.+}} 'char[10]' lvalue Var {{.+}} 'local' 'char[10]'
// CHECK-NEXT: |     |     `-IntegerLiteral {{.+}} 'int' 2
// CHECK-NEXT: |     `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |       `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |         `-UnaryOperator {{.+}} 'char *__bidi_indexable' prefix '&' cannot overflow
// CHECK-NEXT: |           `-ArraySubscriptExpr {{.+}} 'char' lvalue
// CHECK-NEXT: |             |-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: |             | `-DeclRefExpr {{.+}} 'char[10]' lvalue Var {{.+}} 'local' 'char[10]'
// CHECK-NEXT: |             `-IntegerLiteral {{.+}} 'int' 10
void pass_const_size_arr_start_oob(void) {
  char local[10];
  ended_by(local - 2, &local[10]);
}

// CHECK-LABEL:|-FunctionDecl {{.+}} pass_const_size_arr_end_oob 'void (void)'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   |-DeclStmt {{.+}}
// CHECK-NEXT: |   | `-VarDecl {{.+}} used local 'char[10]'
// CHECK-NEXT: |   `-MaterializeSequenceExpr {{.+}} 'void' <Unbind>
// CHECK-NEXT: |     |-MaterializeSequenceExpr {{.+}} 'void' <Bind>
// CHECK-NEXT: |     | |-BoundsCheckExpr {{.+}} 'void' 'local + 11 <= __builtin_get_pointer_upper_bound(local) && local <= local + 11 && __builtin_get_pointer_lower_bound(local) <= local'
// CHECK-NEXT: |     | | |-CallExpr {{.+}} 'void'
// CHECK-NEXT: |     | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(const char *__single __ended_by(end), const char *__single /* __started_by(start) */ )' <FunctionToPointerDecay>
// CHECK-NEXT: |     | | | | `-DeclRefExpr {{.+}} 'void (const char *__single __ended_by(end), const char *__single /* __started_by(start) */ )' Function {{.+}} 'ended_by' 'void (const char *__single __ended_by(end), const char *__single /* __started_by(start) */ )'
// CHECK-NEXT: |     | | | |-ImplicitCastExpr {{.+}} 'const char *__single __ended_by(end)':'const char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | | | | `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | | | |   `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | | | |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: |     | | | |       `-DeclRefExpr {{.+}} 'char[10]' lvalue Var {{.+}} 'local' 'char[10]'
// CHECK-NEXT: |     | | | `-ImplicitCastExpr {{.+}} 'const char *__single /* __started_by(start) */ ':'const char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | | |   `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | | |     `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | | |       `-BinaryOperator {{.+}} 'char *__bidi_indexable' '+'
// CHECK-NEXT: |     | | |         |-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: |     | | |         | `-DeclRefExpr {{.+}} 'char[10]' lvalue Var {{.+}} 'local' 'char[10]'
// CHECK-NEXT: |     | | |         `-IntegerLiteral {{.+}} 'int' 11
// CHECK-NEXT: |     | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |     | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |     | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |     | |   | | |-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | |   | | | `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | |   | | |   `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | |   | | |     `-BinaryOperator {{.+}} 'char *__bidi_indexable' '+'
// CHECK-NEXT: |     | |   | | |       |-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: |     | |   | | |       | `-DeclRefExpr {{.+}} 'char[10]' lvalue Var {{.+}} 'local' 'char[10]'
// CHECK-NEXT: |     | |   | | |       `-IntegerLiteral {{.+}} 'int' 11
// CHECK-NEXT: |     | |   | | `-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | |   | |   `-GetBoundExpr {{.+}} 'const char *__bidi_indexable' upper
// CHECK-NEXT: |     | |   | |     `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | |   | |       `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | |   | |         `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: |     | |   | |           `-DeclRefExpr {{.+}} 'char[10]' lvalue Var {{.+}} 'local' 'char[10]'
// CHECK-NEXT: |     | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |     | |   |   |-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | |   |   | `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | |   |   |   `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | |   |   |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: |     | |   |   |       `-DeclRefExpr {{.+}} 'char[10]' lvalue Var {{.+}} 'local' 'char[10]'
// CHECK-NEXT: |     | |   |   `-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | |   |     `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | |   |       `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | |   |         `-BinaryOperator {{.+}} 'char *__bidi_indexable' '+'
// CHECK-NEXT: |     | |   |           |-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: |     | |   |           | `-DeclRefExpr {{.+}} 'char[10]' lvalue Var {{.+}} 'local' 'char[10]'
// CHECK-NEXT: |     | |   |           `-IntegerLiteral {{.+}} 'int' 11
// CHECK-NEXT: |     | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |     | |     |-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | |     | `-GetBoundExpr {{.+}} 'const char *__bidi_indexable' lower
// CHECK-NEXT: |     | |     |   `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | |     |     `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | |     |       `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: |     | |     |         `-DeclRefExpr {{.+}} 'char[10]' lvalue Var {{.+}} 'local' 'char[10]'
// CHECK-NEXT: |     | |     `-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | |       `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | |         `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | |           `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: |     | |             `-DeclRefExpr {{.+}} 'char[10]' lvalue Var {{.+}} 'local' 'char[10]'
// CHECK-NEXT: |     | |-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | | `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: |     | |     `-DeclRefExpr {{.+}} 'char[10]' lvalue Var {{.+}} 'local' 'char[10]'
// CHECK-NEXT: |     | `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     |   `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     |     `-BinaryOperator {{.+}} 'char *__bidi_indexable' '+'
// CHECK-NEXT: |     |       |-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: |     |       | `-DeclRefExpr {{.+}} 'char[10]' lvalue Var {{.+}} 'local' 'char[10]'
// CHECK-NEXT: |     |       `-IntegerLiteral {{.+}} 'int' 11
// CHECK-NEXT: |     |-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: |     |     `-DeclRefExpr {{.+}} 'char[10]' lvalue Var {{.+}} 'local' 'char[10]'
// CHECK-NEXT: |     `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |       `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |         `-BinaryOperator {{.+}} 'char *__bidi_indexable' '+'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: |           | `-DeclRefExpr {{.+}} 'char[10]' lvalue Var {{.+}} 'local' 'char[10]'
// CHECK-NEXT: |           `-IntegerLiteral {{.+}} 'int' 11
void pass_const_size_arr_end_oob(void) {
  char local[10];
  ended_by(local, local + 11);
}

// CHECK-LABEL:|-FunctionDecl {{.+}} pass_explicit_indexable 'void (void)'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   |-DeclStmt {{.+}}
// CHECK-NEXT: |   | `-VarDecl {{.+}} used local 'char[10]'
// CHECK-NEXT: |   |-DeclStmt {{.+}}
// CHECK-NEXT: |   | `-VarDecl {{.+}} used ilocal 'char *__indexable' cinit
// CHECK-NEXT: |   |   `-ImplicitCastExpr {{.+}} 'char *__indexable' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: |   |       `-DeclRefExpr {{.+}} 'char[10]' lvalue Var {{.+}} 'local' 'char[10]'
// CHECK-NEXT: |   `-MaterializeSequenceExpr {{.+}} 'void' <Unbind>
// CHECK-NEXT: |     |-MaterializeSequenceExpr {{.+}} 'void' <Bind>
// CHECK-NEXT: |     | |-BoundsCheckExpr {{.+}} 'void' '&ilocal[10] <= __builtin_get_pointer_upper_bound(ilocal) && ilocal <= &ilocal[10] && __builtin_get_pointer_lower_bound(ilocal) <= ilocal'
// CHECK-NEXT: |     | | |-CallExpr {{.+}} 'void'
// CHECK-NEXT: |     | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(const char *__single __ended_by(end), const char *__single /* __started_by(start) */ )' <FunctionToPointerDecay>
// CHECK-NEXT: |     | | | | `-DeclRefExpr {{.+}} 'void (const char *__single __ended_by(end), const char *__single /* __started_by(start) */ )' Function {{.+}} 'ended_by' 'void (const char *__single __ended_by(end), const char *__single /* __started_by(start) */ )'
// CHECK-NEXT: |     | | | |-ImplicitCastExpr {{.+}} 'const char *__single __ended_by(end)':'const char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | | | | `-OpaqueValueExpr {{.+}} 'const char *__indexable'
// CHECK-NEXT: |     | | | |   `-ImplicitCastExpr {{.+}} 'const char *__indexable' <NoOp>
// CHECK-NEXT: |     | | | |     `-ImplicitCastExpr {{.+}} 'char *__indexable' <LValueToRValue>
// CHECK-NEXT: |     | | | |       `-DeclRefExpr {{.+}} 'char *__indexable' lvalue Var {{.+}} 'ilocal' 'char *__indexable'
// CHECK-NEXT: |     | | | `-ImplicitCastExpr {{.+}} 'const char *__single /* __started_by(start) */ ':'const char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | | |   `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | | |     `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | | |       `-UnaryOperator {{.+}} 'char *__bidi_indexable' prefix '&' cannot overflow
// CHECK-NEXT: |     | | |         `-ArraySubscriptExpr {{.+}} 'char' lvalue
// CHECK-NEXT: |     | | |           |-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | | |           | `-ImplicitCastExpr {{.+}} 'char *__indexable' <LValueToRValue>
// CHECK-NEXT: |     | | |           |   `-DeclRefExpr {{.+}} 'char *__indexable' lvalue Var {{.+}} 'ilocal' 'char *__indexable'
// CHECK-NEXT: |     | | |           `-IntegerLiteral {{.+}} 'int' 10
// CHECK-NEXT: |     | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |     | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |     | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |     | |   | | |-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | |   | | | `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | |   | | |   `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | |   | | |     `-UnaryOperator {{.+}} 'char *__bidi_indexable' prefix '&' cannot overflow
// CHECK-NEXT: |     | |   | | |       `-ArraySubscriptExpr {{.+}} 'char' lvalue
// CHECK-NEXT: |     | |   | | |         |-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | |   | | |         | `-ImplicitCastExpr {{.+}} 'char *__indexable' <LValueToRValue>
// CHECK-NEXT: |     | |   | | |         |   `-DeclRefExpr {{.+}} 'char *__indexable' lvalue Var {{.+}} 'ilocal' 'char *__indexable'
// CHECK-NEXT: |     | |   | | |         `-IntegerLiteral {{.+}} 'int' 10
// CHECK-NEXT: |     | |   | | `-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | |   | |   `-GetBoundExpr {{.+}} 'const char *__bidi_indexable' upper
// CHECK-NEXT: |     | |   | |     `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | |   | |       `-OpaqueValueExpr {{.+}} 'const char *__indexable'
// CHECK-NEXT: |     | |   | |         `-ImplicitCastExpr {{.+}} 'const char *__indexable' <NoOp>
// CHECK-NEXT: |     | |   | |           `-ImplicitCastExpr {{.+}} 'char *__indexable' <LValueToRValue>
// CHECK-NEXT: |     | |   | |             `-DeclRefExpr {{.+}} 'char *__indexable' lvalue Var {{.+}} 'ilocal' 'char *__indexable'
// CHECK-NEXT: |     | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |     | |   |   |-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | |   |   | `-OpaqueValueExpr {{.+}} 'const char *__indexable'
// CHECK-NEXT: |     | |   |   |   `-ImplicitCastExpr {{.+}} 'const char *__indexable' <NoOp>
// CHECK-NEXT: |     | |   |   |     `-ImplicitCastExpr {{.+}} 'char *__indexable' <LValueToRValue>
// CHECK-NEXT: |     | |   |   |       `-DeclRefExpr {{.+}} 'char *__indexable' lvalue Var {{.+}} 'ilocal' 'char *__indexable'
// CHECK-NEXT: |     | |   |   `-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | |   |     `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | |   |       `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | |   |         `-UnaryOperator {{.+}} 'char *__bidi_indexable' prefix '&' cannot overflow
// CHECK-NEXT: |     | |   |           `-ArraySubscriptExpr {{.+}} 'char' lvalue
// CHECK-NEXT: |     | |   |             |-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | |   |             | `-ImplicitCastExpr {{.+}} 'char *__indexable' <LValueToRValue>
// CHECK-NEXT: |     | |   |             |   `-DeclRefExpr {{.+}} 'char *__indexable' lvalue Var {{.+}} 'ilocal' 'char *__indexable'
// CHECK-NEXT: |     | |   |             `-IntegerLiteral {{.+}} 'int' 10
// CHECK-NEXT: |     | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |     | |     |-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | |     | `-GetBoundExpr {{.+}} 'const char *__bidi_indexable' lower
// CHECK-NEXT: |     | |     |   `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | |     |     `-OpaqueValueExpr {{.+}} 'const char *__indexable'
// CHECK-NEXT: |     | |     |       `-ImplicitCastExpr {{.+}} 'const char *__indexable' <NoOp>
// CHECK-NEXT: |     | |     |         `-ImplicitCastExpr {{.+}} 'char *__indexable' <LValueToRValue>
// CHECK-NEXT: |     | |     |           `-DeclRefExpr {{.+}} 'char *__indexable' lvalue Var {{.+}} 'ilocal' 'char *__indexable'
// CHECK-NEXT: |     | |     `-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | |       `-OpaqueValueExpr {{.+}} 'const char *__indexable'
// CHECK-NEXT: |     | |         `-ImplicitCastExpr {{.+}} 'const char *__indexable' <NoOp>
// CHECK-NEXT: |     | |           `-ImplicitCastExpr {{.+}} 'char *__indexable' <LValueToRValue>
// CHECK-NEXT: |     | |             `-DeclRefExpr {{.+}} 'char *__indexable' lvalue Var {{.+}} 'ilocal' 'char *__indexable'
// CHECK-NEXT: |     | |-OpaqueValueExpr {{.+}} 'const char *__indexable'
// CHECK-NEXT: |     | | `-ImplicitCastExpr {{.+}} 'const char *__indexable' <NoOp>
// CHECK-NEXT: |     | |   `-ImplicitCastExpr {{.+}} 'char *__indexable' <LValueToRValue>
// CHECK-NEXT: |     | |     `-DeclRefExpr {{.+}} 'char *__indexable' lvalue Var {{.+}} 'ilocal' 'char *__indexable'
// CHECK-NEXT: |     | `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     |   `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     |     `-UnaryOperator {{.+}} 'char *__bidi_indexable' prefix '&' cannot overflow
// CHECK-NEXT: |     |       `-ArraySubscriptExpr {{.+}} 'char' lvalue
// CHECK-NEXT: |     |         |-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     |         | `-ImplicitCastExpr {{.+}} 'char *__indexable' <LValueToRValue>
// CHECK-NEXT: |     |         |   `-DeclRefExpr {{.+}} 'char *__indexable' lvalue Var {{.+}} 'ilocal' 'char *__indexable'
// CHECK-NEXT: |     |         `-IntegerLiteral {{.+}} 'int' 10
// CHECK-NEXT: |     |-OpaqueValueExpr {{.+}} 'const char *__indexable'
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'const char *__indexable' <NoOp>
// CHECK-NEXT: |     |   `-ImplicitCastExpr {{.+}} 'char *__indexable' <LValueToRValue>
// CHECK-NEXT: |     |     `-DeclRefExpr {{.+}} 'char *__indexable' lvalue Var {{.+}} 'ilocal' 'char *__indexable'
// CHECK-NEXT: |     `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |       `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |         `-UnaryOperator {{.+}} 'char *__bidi_indexable' prefix '&' cannot overflow
// CHECK-NEXT: |           `-ArraySubscriptExpr {{.+}} 'char' lvalue
// CHECK-NEXT: |             |-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK-NEXT: |             | `-ImplicitCastExpr {{.+}} 'char *__indexable' <LValueToRValue>
// CHECK-NEXT: |             |   `-DeclRefExpr {{.+}} 'char *__indexable' lvalue Var {{.+}} 'ilocal' 'char *__indexable'
// CHECK-NEXT: |             `-IntegerLiteral {{.+}} 'int' 10
void pass_explicit_indexable(void) {
  char local[10];
  char* __indexable ilocal = local;
  ended_by(ilocal, &ilocal[10]);
}

// CHECK-LABEL:|-FunctionDecl {{.+}} pass_explict_bidi_indexable 'void (void)'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   |-DeclStmt {{.+}}
// CHECK-NEXT: |   | `-VarDecl {{.+}} used local 'char[10]'
// CHECK-NEXT: |   |-DeclStmt {{.+}}
// CHECK-NEXT: |   | `-VarDecl {{.+}} used bilocal 'char *__bidi_indexable' cinit
// CHECK-NEXT: |   |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: |   |     `-DeclRefExpr {{.+}} 'char[10]' lvalue Var {{.+}} 'local' 'char[10]'
// CHECK-NEXT: |   `-MaterializeSequenceExpr {{.+}} 'void' <Unbind>
// CHECK-NEXT: |     |-MaterializeSequenceExpr {{.+}} 'void' <Bind>
// CHECK-NEXT: |     | |-BoundsCheckExpr {{.+}} 'void' '&bilocal[10] <= __builtin_get_pointer_upper_bound(bilocal) && bilocal <= &bilocal[10] && __builtin_get_pointer_lower_bound(bilocal) <= bilocal'
// CHECK-NEXT: |     | | |-CallExpr {{.+}} 'void'
// CHECK-NEXT: |     | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(const char *__single __ended_by(end), const char *__single /* __started_by(start) */ )' <FunctionToPointerDecay>
// CHECK-NEXT: |     | | | | `-DeclRefExpr {{.+}} 'void (const char *__single __ended_by(end), const char *__single /* __started_by(start) */ )' Function {{.+}} 'ended_by' 'void (const char *__single __ended_by(end), const char *__single /* __started_by(start) */ )'
// CHECK-NEXT: |     | | | |-ImplicitCastExpr {{.+}} 'const char *__single __ended_by(end)':'const char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | | | | `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | | | |   `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | | | |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |     | | | |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue Var {{.+}} 'bilocal' 'char *__bidi_indexable'
// CHECK-NEXT: |     | | | `-ImplicitCastExpr {{.+}} 'const char *__single /* __started_by(start) */ ':'const char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | | |   `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | | |     `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | | |       `-UnaryOperator {{.+}} 'char *__bidi_indexable' prefix '&' cannot overflow
// CHECK-NEXT: |     | | |         `-ArraySubscriptExpr {{.+}} 'char' lvalue
// CHECK-NEXT: |     | | |           |-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |     | | |           | `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue Var {{.+}} 'bilocal' 'char *__bidi_indexable'
// CHECK-NEXT: |     | | |           `-IntegerLiteral {{.+}} 'int' 10
// CHECK-NEXT: |     | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |     | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |     | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |     | |   | | |-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | |   | | | `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | |   | | |   `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | |   | | |     `-UnaryOperator {{.+}} 'char *__bidi_indexable' prefix '&' cannot overflow
// CHECK-NEXT: |     | |   | | |       `-ArraySubscriptExpr {{.+}} 'char' lvalue
// CHECK-NEXT: |     | |   | | |         |-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |     | |   | | |         | `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue Var {{.+}} 'bilocal' 'char *__bidi_indexable'
// CHECK-NEXT: |     | |   | | |         `-IntegerLiteral {{.+}} 'int' 10
// CHECK-NEXT: |     | |   | | `-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | |   | |   `-GetBoundExpr {{.+}} 'const char *__bidi_indexable' upper
// CHECK-NEXT: |     | |   | |     `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | |   | |       `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | |   | |         `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |     | |   | |           `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue Var {{.+}} 'bilocal' 'char *__bidi_indexable'
// CHECK-NEXT: |     | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |     | |   |   |-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | |   |   | `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | |   |   |   `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | |   |   |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |     | |   |   |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue Var {{.+}} 'bilocal' 'char *__bidi_indexable'
// CHECK-NEXT: |     | |   |   `-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | |   |     `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | |   |       `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | |   |         `-UnaryOperator {{.+}} 'char *__bidi_indexable' prefix '&' cannot overflow
// CHECK-NEXT: |     | |   |           `-ArraySubscriptExpr {{.+}} 'char' lvalue
// CHECK-NEXT: |     | |   |             |-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |     | |   |             | `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue Var {{.+}} 'bilocal' 'char *__bidi_indexable'
// CHECK-NEXT: |     | |   |             `-IntegerLiteral {{.+}} 'int' 10
// CHECK-NEXT: |     | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |     | |     |-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | |     | `-GetBoundExpr {{.+}} 'const char *__bidi_indexable' lower
// CHECK-NEXT: |     | |     |   `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | |     |     `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | |     |       `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |     | |     |         `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue Var {{.+}} 'bilocal' 'char *__bidi_indexable'
// CHECK-NEXT: |     | |     `-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | |       `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | |         `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | |           `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |     | |             `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue Var {{.+}} 'bilocal' 'char *__bidi_indexable'
// CHECK-NEXT: |     | |-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | | `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |     | |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue Var {{.+}} 'bilocal' 'char *__bidi_indexable'
// CHECK-NEXT: |     | `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     |   `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     |     `-UnaryOperator {{.+}} 'char *__bidi_indexable' prefix '&' cannot overflow
// CHECK-NEXT: |     |       `-ArraySubscriptExpr {{.+}} 'char' lvalue
// CHECK-NEXT: |     |         |-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |     |         | `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue Var {{.+}} 'bilocal' 'char *__bidi_indexable'
// CHECK-NEXT: |     |         `-IntegerLiteral {{.+}} 'int' 10
// CHECK-NEXT: |     |-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |     |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue Var {{.+}} 'bilocal' 'char *__bidi_indexable'
// CHECK-NEXT: |     `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |       `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |         `-UnaryOperator {{.+}} 'char *__bidi_indexable' prefix '&' cannot overflow
// CHECK-NEXT: |           `-ArraySubscriptExpr {{.+}} 'char' lvalue
// CHECK-NEXT: |             |-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |             | `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue Var {{.+}} 'bilocal' 'char *__bidi_indexable'
// CHECK-NEXT: |             `-IntegerLiteral {{.+}} 'int' 10
void pass_explict_bidi_indexable(void) {
  char local[10];
  char* __bidi_indexable bilocal = local;
  ended_by(bilocal, &bilocal[10]);
}

// CHECK-LABEL:|-FunctionDecl {{.+}} pass_ended_by 'void (char *__single __ended_by(end), const char *__single /* __started_by(start) */ )'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used start 'char *__single __ended_by(end)':'char *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used end 'const char *__single /* __started_by(start) */ ':'const char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-MaterializeSequenceExpr {{.+}} 'void' <Unbind>
// CHECK-NEXT: |     |-MaterializeSequenceExpr {{.+}} 'void' <Bind>
// CHECK-NEXT: |     | |-BoundsCheckExpr {{.+}} 'void' 'end <= __builtin_get_pointer_upper_bound(start) && start <= end && __builtin_get_pointer_lower_bound(start) <= start'
// CHECK-NEXT: |     | | |-CallExpr {{.+}} 'void'
// CHECK-NEXT: |     | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(const char *__single __ended_by(end), const char *__single /* __started_by(start) */ )' <FunctionToPointerDecay>
// CHECK-NEXT: |     | | | | `-DeclRefExpr {{.+}} 'void (const char *__single __ended_by(end), const char *__single /* __started_by(start) */ )' Function {{.+}} 'ended_by' 'void (const char *__single __ended_by(end), const char *__single /* __started_by(start) */ )'
// CHECK-NEXT: |     | | | |-ImplicitCastExpr {{.+}} 'const char *__single __ended_by(end)':'const char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | | | | `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | | | |   `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | | | |     `-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |     | | | |       |-DeclRefExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __ended_by(end)':'char *__single'
// CHECK-NEXT: |     | | | |       |-ImplicitCastExpr {{.+}} 'const char *__single /* __started_by(start) */ ':'const char *__single' <LValueToRValue>
// CHECK-NEXT: |     | | | |       | `-DeclRefExpr {{.+}} 'const char *__single /* __started_by(start) */ ':'const char *__single' lvalue ParmVar {{.+}} 'end' 'const char *__single /* __started_by(start) */ ':'const char *__single'
// CHECK-NEXT: |     | | | |       `-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |     | | | |         `-DeclRefExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __ended_by(end)':'char *__single'
// CHECK-NEXT: |     | | | `-ImplicitCastExpr {{.+}} 'const char *__single /* __started_by(start) */ ':'const char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | | |   `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | | |     `-BoundsSafetyPointerPromotionExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | | |       |-DeclRefExpr {{.+}} 'const char *__single /* __started_by(start) */ ':'const char *__single' lvalue ParmVar {{.+}} 'end' 'const char *__single /* __started_by(start) */ ':'const char *__single'
// CHECK-NEXT: |     | | |       |-ImplicitCastExpr {{.+}} 'const char *__single /* __started_by(start) */ ':'const char *__single' <LValueToRValue>
// CHECK-NEXT: |     | | |       | `-DeclRefExpr {{.+}} 'const char *__single /* __started_by(start) */ ':'const char *__single' lvalue ParmVar {{.+}} 'end' 'const char *__single /* __started_by(start) */ ':'const char *__single'
// CHECK-NEXT: |     | | |       `-ImplicitCastExpr {{.+}} <<invalid sloc>> 'char *__single __ended_by(end)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |     | | |         `-DeclRefExpr {{.+}} <<invalid sloc>> 'char *__single __ended_by(end)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __ended_by(end)':'char *__single'
// CHECK-NEXT: |     | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |     | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |     | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |     | |   | | |-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | |   | | | `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | |   | | |   `-BoundsSafetyPointerPromotionExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | |   | | |     |-DeclRefExpr {{.+}} 'const char *__single /* __started_by(start) */ ':'const char *__single' lvalue ParmVar {{.+}} 'end' 'const char *__single /* __started_by(start) */ ':'const char *__single'
// CHECK-NEXT: |     | |   | | |     |-ImplicitCastExpr {{.+}} 'const char *__single /* __started_by(start) */ ':'const char *__single' <LValueToRValue>
// CHECK-NEXT: |     | |   | | |     | `-DeclRefExpr {{.+}} 'const char *__single /* __started_by(start) */ ':'const char *__single' lvalue ParmVar {{.+}} 'end' 'const char *__single /* __started_by(start) */ ':'const char *__single'
// CHECK-NEXT: |     | |   | | |     `-ImplicitCastExpr {{.+}} <<invalid sloc>> 'char *__single __ended_by(end)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |     | |   | | |       `-DeclRefExpr {{.+}} <<invalid sloc>> 'char *__single __ended_by(end)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __ended_by(end)':'char *__single'
// CHECK-NEXT: |     | |   | | `-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | |   | |   `-GetBoundExpr {{.+}} 'const char *__bidi_indexable' upper
// CHECK-NEXT: |     | |   | |     `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | |   | |       `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | |   | |         `-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |     | |   | |           |-DeclRefExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __ended_by(end)':'char *__single'
// CHECK-NEXT: |     | |   | |           |-ImplicitCastExpr {{.+}} 'const char *__single /* __started_by(start) */ ':'const char *__single' <LValueToRValue>
// CHECK-NEXT: |     | |   | |           | `-DeclRefExpr {{.+}} 'const char *__single /* __started_by(start) */ ':'const char *__single' lvalue ParmVar {{.+}} 'end' 'const char *__single /* __started_by(start) */ ':'const char *__single'
// CHECK-NEXT: |     | |   | |           `-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |     | |   | |             `-DeclRefExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __ended_by(end)':'char *__single'
// CHECK-NEXT: |     | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |     | |   |   |-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | |   |   | `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | |   |   |   `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | |   |   |     `-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |     | |   |   |       |-DeclRefExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __ended_by(end)':'char *__single'
// CHECK-NEXT: |     | |   |   |       |-ImplicitCastExpr {{.+}} 'const char *__single /* __started_by(start) */ ':'const char *__single' <LValueToRValue>
// CHECK-NEXT: |     | |   |   |       | `-DeclRefExpr {{.+}} 'const char *__single /* __started_by(start) */ ':'const char *__single' lvalue ParmVar {{.+}} 'end' 'const char *__single /* __started_by(start) */ ':'const char *__single'
// CHECK-NEXT: |     | |   |   |       `-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |     | |   |   |         `-DeclRefExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __ended_by(end)':'char *__single'
// CHECK-NEXT: |     | |   |   `-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | |   |     `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | |   |       `-BoundsSafetyPointerPromotionExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | |   |         |-DeclRefExpr {{.+}} 'const char *__single /* __started_by(start) */ ':'const char *__single' lvalue ParmVar {{.+}} 'end' 'const char *__single /* __started_by(start) */ ':'const char *__single'
// CHECK-NEXT: |     | |   |         |-ImplicitCastExpr {{.+}} 'const char *__single /* __started_by(start) */ ':'const char *__single' <LValueToRValue>
// CHECK-NEXT: |     | |   |         | `-DeclRefExpr {{.+}} 'const char *__single /* __started_by(start) */ ':'const char *__single' lvalue ParmVar {{.+}} 'end' 'const char *__single /* __started_by(start) */ ':'const char *__single'
// CHECK-NEXT: |     | |   |         `-ImplicitCastExpr {{.+}} <<invalid sloc>> 'char *__single __ended_by(end)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |     | |   |           `-DeclRefExpr {{.+}} <<invalid sloc>> 'char *__single __ended_by(end)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __ended_by(end)':'char *__single'
// CHECK-NEXT: |     | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |     | |     |-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | |     | `-GetBoundExpr {{.+}} 'const char *__bidi_indexable' lower
// CHECK-NEXT: |     | |     |   `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | |     |     `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | |     |       `-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |     | |     |         |-DeclRefExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __ended_by(end)':'char *__single'
// CHECK-NEXT: |     | |     |         |-ImplicitCastExpr {{.+}} 'const char *__single /* __started_by(start) */ ':'const char *__single' <LValueToRValue>
// CHECK-NEXT: |     | |     |         | `-DeclRefExpr {{.+}} 'const char *__single /* __started_by(start) */ ':'const char *__single' lvalue ParmVar {{.+}} 'end' 'const char *__single /* __started_by(start) */ ':'const char *__single'
// CHECK-NEXT: |     | |     |         `-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |     | |     |           `-DeclRefExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __ended_by(end)':'char *__single'
// CHECK-NEXT: |     | |     `-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | |       `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | |         `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | |           `-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |     | |             |-DeclRefExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __ended_by(end)':'char *__single'
// CHECK-NEXT: |     | |             |-ImplicitCastExpr {{.+}} 'const char *__single /* __started_by(start) */ ':'const char *__single' <LValueToRValue>
// CHECK-NEXT: |     | |             | `-DeclRefExpr {{.+}} 'const char *__single /* __started_by(start) */ ':'const char *__single' lvalue ParmVar {{.+}} 'end' 'const char *__single /* __started_by(start) */ ':'const char *__single'
// CHECK-NEXT: |     | |             `-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |     | |               `-DeclRefExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __ended_by(end)':'char *__single'
// CHECK-NEXT: |     | |-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | | `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     | |   `-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |     | |     |-DeclRefExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __ended_by(end)':'char *__single'
// CHECK-NEXT: |     | |     |-ImplicitCastExpr {{.+}} 'const char *__single /* __started_by(start) */ ':'const char *__single' <LValueToRValue>
// CHECK-NEXT: |     | |     | `-DeclRefExpr {{.+}} 'const char *__single /* __started_by(start) */ ':'const char *__single' lvalue ParmVar {{.+}} 'end' 'const char *__single /* __started_by(start) */ ':'const char *__single'
// CHECK-NEXT: |     | |     `-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |     | |       `-DeclRefExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __ended_by(end)':'char *__single'
// CHECK-NEXT: |     | `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     |   `-BoundsSafetyPointerPromotionExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     |     |-DeclRefExpr {{.+}} 'const char *__single /* __started_by(start) */ ':'const char *__single' lvalue ParmVar {{.+}} 'end' 'const char *__single /* __started_by(start) */ ':'const char *__single'
// CHECK-NEXT: |     |     |-ImplicitCastExpr {{.+}} 'const char *__single /* __started_by(start) */ ':'const char *__single' <LValueToRValue>
// CHECK-NEXT: |     |     | `-DeclRefExpr {{.+}} 'const char *__single /* __started_by(start) */ ':'const char *__single' lvalue ParmVar {{.+}} 'end' 'const char *__single /* __started_by(start) */ ':'const char *__single'
// CHECK-NEXT: |     |     `-ImplicitCastExpr {{.+}} <<invalid sloc>> 'char *__single __ended_by(end)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |     |       `-DeclRefExpr {{.+}} <<invalid sloc>> 'char *__single __ended_by(end)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __ended_by(end)':'char *__single'
// CHECK-NEXT: |     |-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |     |   `-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |     |     |-DeclRefExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __ended_by(end)':'char *__single'
// CHECK-NEXT: |     |     |-ImplicitCastExpr {{.+}} 'const char *__single /* __started_by(start) */ ':'const char *__single' <LValueToRValue>
// CHECK-NEXT: |     |     | `-DeclRefExpr {{.+}} 'const char *__single /* __started_by(start) */ ':'const char *__single' lvalue ParmVar {{.+}} 'end' 'const char *__single /* __started_by(start) */ ':'const char *__single'
// CHECK-NEXT: |     |     `-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |     |       `-DeclRefExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __ended_by(end)':'char *__single'
// CHECK-NEXT: |     `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |       `-BoundsSafetyPointerPromotionExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT: |         |-DeclRefExpr {{.+}} 'const char *__single /* __started_by(start) */ ':'const char *__single' lvalue ParmVar {{.+}} 'end' 'const char *__single /* __started_by(start) */ ':'const char *__single'
// CHECK-NEXT: |         |-ImplicitCastExpr {{.+}} 'const char *__single /* __started_by(start) */ ':'const char *__single' <LValueToRValue>
// CHECK-NEXT: |         | `-DeclRefExpr {{.+}} 'const char *__single /* __started_by(start) */ ':'const char *__single' lvalue ParmVar {{.+}} 'end' 'const char *__single /* __started_by(start) */ ':'const char *__single'
// CHECK-NEXT: |         `-ImplicitCastExpr {{.+}} <<invalid sloc>> 'char *__single __ended_by(end)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           `-DeclRefExpr {{.+}} <<invalid sloc>> 'char *__single __ended_by(end)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __ended_by(end)':'char *__single'
void pass_ended_by(char* __ended_by(end) start, const char* end) {
  ended_by(start, end);
}

// CHECK-LABEL:`-FunctionDecl {{.+}} pass_counted_by 'void (char *__single __counted_by(count), int)'
// CHECK-NEXT:   |-ParmVarDecl {{.+}} used start 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:   |-ParmVarDecl {{.+}} used count 'int'
// CHECK-NEXT:   | `-DependerDeclsAttr {{.+}} <<invalid sloc>> Implicit {{.+}} 0
// CHECK-NEXT:   `-CompoundStmt {{.+}}
// CHECK-NEXT:     `-MaterializeSequenceExpr {{.+}} 'void' <Unbind>
// CHECK-NEXT:       |-MaterializeSequenceExpr {{.+}} 'void' <Bind>
// CHECK-NEXT:       | |-BoundsCheckExpr {{.+}} 'void' 'start + count <= __builtin_get_pointer_upper_bound(start) && start <= start + count && __builtin_get_pointer_lower_bound(start) <= start'
// CHECK-NEXT:       | | |-CallExpr {{.+}} 'void'
// CHECK-NEXT:       | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(const char *__single __ended_by(end), const char *__single /* __started_by(start) */ )' <FunctionToPointerDecay>
// CHECK-NEXT:       | | | | `-DeclRefExpr {{.+}} 'void (const char *__single __ended_by(end), const char *__single /* __started_by(start) */ )' Function {{.+}} 'ended_by' 'void (const char *__single __ended_by(end), const char *__single /* __started_by(start) */ )'
// CHECK-NEXT:       | | | |-ImplicitCastExpr {{.+}} 'const char *__single __ended_by(end)':'const char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT:       | | | | `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT:       | | | |   `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT:       | | | |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT:       | | | |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT:       | | | |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT:       | | | |       | | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | | | |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:       | | | |       | | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | | | |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT:       | | | |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT:       | | | |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | | | |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:       | | | |       | | | |     `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | | | |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | | | |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:       | | | |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count' 'int'
// CHECK-NEXT:       | | | |       | | `-<<<NULL>>>
// CHECK-NEXT:       | | | |       | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | | | |       | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:       | | | |       | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | | | |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | | | |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:       | | | |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count' 'int'
// CHECK-NEXT:       | | | |       |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | | | |       | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:       | | | |       |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | | | |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | | | |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:       | | | |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count' 'int'
// CHECK-NEXT:       | | | `-ImplicitCastExpr {{.+}} 'const char *__single /* __started_by(start) */ ':'const char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT:       | | |   `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT:       | | |     `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT:       | | |       `-BinaryOperator {{.+}} 'char *__bidi_indexable' '+'
// CHECK-NEXT:       | | |         |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT:       | | |         | |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT:       | | |         | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT:       | | |         | | | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | | |         | | | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:       | | |         | | | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | | |         | | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT:       | | |         | | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT:       | | |         | | | | | `-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | | |         | | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:       | | |         | | | | |     `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | | |         | | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | | |         | | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:       | | |         | | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count' 'int'
// CHECK-NEXT:       | | |         | | | `-<<<NULL>>>
// CHECK-NEXT:       | | |         | | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | | |         | | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:       | | |         | | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | | |         | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | | |         | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:       | | |         | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count' 'int'
// CHECK-NEXT:       | | |         | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | | |         | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:       | | |         | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | | |         | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | | |         |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:       | | |         |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count' 'int'
// CHECK-NEXT:       | | |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:       | | |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count' 'int'
// CHECK-NEXT:       | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT:       | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT:       | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT:       | |   | | |-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK-NEXT:       | |   | | | `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT:       | |   | | |   `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT:       | |   | | |     `-BinaryOperator {{.+}} 'char *__bidi_indexable' '+'
// CHECK-NEXT:       | |   | | |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT:       | |   | | |       | |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT:       | |   | | |       | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT:       | |   | | |       | | | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |   | | |       | | | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:       | |   | | |       | | | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |   | | |       | | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT:       | |   | | |       | | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT:       | |   | | |       | | | | | `-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |   | | |       | | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:       | |   | | |       | | | | |     `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |   | | |       | | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | |   | | |       | | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:       | |   | | |       | | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count' 'int'
// CHECK-NEXT:       | |   | | |       | | | `-<<<NULL>>>
// CHECK-NEXT:       | |   | | |       | | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |   | | |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:       | |   | | |       | | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |   | | |       | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | |   | | |       | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:       | |   | | |       | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count' 'int'
// CHECK-NEXT:       | |   | | |       | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |   | | |       | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:       | |   | | |       | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |   | | |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | |   | | |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:       | |   | | |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count' 'int'
// CHECK-NEXT:       | |   | | |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:       | |   | | |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count' 'int'
// CHECK-NEXT:       | |   | | `-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK-NEXT:       | |   | |   `-GetBoundExpr {{.+}} 'const char *__bidi_indexable' upper
// CHECK-NEXT:       | |   | |     `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT:       | |   | |       `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT:       | |   | |         `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT:       | |   | |           |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT:       | |   | |           | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT:       | |   | |           | | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |   | |           | | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:       | |   | |           | | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |   | |           | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT:       | |   | |           | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT:       | |   | |           | | | | `-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |   | |           | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:       | |   | |           | | | |     `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |   | |           | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | |   | |           | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:       | |   | |           | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count' 'int'
// CHECK-NEXT:       | |   | |           | | `-<<<NULL>>>
// CHECK-NEXT:       | |   | |           | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |   | |           | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:       | |   | |           | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |   | |           | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | |   | |           |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:       | |   | |           |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count' 'int'
// CHECK-NEXT:       | |   | |           |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |   | |           | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:       | |   | |           |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |   | |           `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | |   | |             `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:       | |   | |               `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count' 'int'
// CHECK-NEXT:       | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT:       | |   |   |-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK-NEXT:       | |   |   | `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT:       | |   |   |   `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT:       | |   |   |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT:       | |   |   |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT:       | |   |   |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT:       | |   |   |       | | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |   |   |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:       | |   |   |       | | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |   |   |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT:       | |   |   |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT:       | |   |   |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |   |   |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:       | |   |   |       | | | |     `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |   |   |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | |   |   |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:       | |   |   |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count' 'int'
// CHECK-NEXT:       | |   |   |       | | `-<<<NULL>>>
// CHECK-NEXT:       | |   |   |       | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |   |   |       | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:       | |   |   |       | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |   |   |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | |   |   |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:       | |   |   |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count' 'int'
// CHECK-NEXT:       | |   |   |       |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |   |   |       | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:       | |   |   |       |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |   |   |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | |   |   |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:       | |   |   |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count' 'int'
// CHECK-NEXT:       | |   |   `-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK-NEXT:       | |   |     `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT:       | |   |       `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT:       | |   |         `-BinaryOperator {{.+}} 'char *__bidi_indexable' '+'
// CHECK-NEXT:       | |   |           |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT:       | |   |           | |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT:       | |   |           | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT:       | |   |           | | | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |   |           | | | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:       | |   |           | | | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |   |           | | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT:       | |   |           | | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT:       | |   |           | | | | | `-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |   |           | | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:       | |   |           | | | | |     `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |   |           | | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | |   |           | | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:       | |   |           | | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count' 'int'
// CHECK-NEXT:       | |   |           | | | `-<<<NULL>>>
// CHECK-NEXT:       | |   |           | | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |   |           | | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:       | |   |           | | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |   |           | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | |   |           | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:       | |   |           | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count' 'int'
// CHECK-NEXT:       | |   |           | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |   |           | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:       | |   |           | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |   |           | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | |   |           |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:       | |   |           |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count' 'int'
// CHECK-NEXT:       | |   |           `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:       | |   |             `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count' 'int'
// CHECK-NEXT:       | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT:       | |     |-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK-NEXT:       | |     | `-GetBoundExpr {{.+}} 'const char *__bidi_indexable' lower
// CHECK-NEXT:       | |     |   `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT:       | |     |     `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT:       | |     |       `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT:       | |     |         |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT:       | |     |         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT:       | |     |         | | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |     |         | | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:       | |     |         | | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |     |         | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT:       | |     |         | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT:       | |     |         | | | | `-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |     |         | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:       | |     |         | | | |     `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |     |         | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | |     |         | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:       | |     |         | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count' 'int'
// CHECK-NEXT:       | |     |         | | `-<<<NULL>>>
// CHECK-NEXT:       | |     |         | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |     |         | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:       | |     |         | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |     |         | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | |     |         |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:       | |     |         |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count' 'int'
// CHECK-NEXT:       | |     |         |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |     |         | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:       | |     |         |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |     |         `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | |     |           `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:       | |     |             `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count' 'int'
// CHECK-NEXT:       | |     `-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK-NEXT:       | |       `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT:       | |         `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT:       | |           `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT:       | |             |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT:       | |             | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT:       | |             | | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |             | | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:       | |             | | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |             | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT:       | |             | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT:       | |             | | | | `-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |             | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:       | |             | | | |     `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |             | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | |             | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:       | |             | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count' 'int'
// CHECK-NEXT:       | |             | | `-<<<NULL>>>
// CHECK-NEXT:       | |             | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |             | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:       | |             | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |             | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | |             |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:       | |             |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count' 'int'
// CHECK-NEXT:       | |             |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |             | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:       | |             |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |             `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | |               `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:       | |                 `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count' 'int'
// CHECK-NEXT:       | |-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT:       | | `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT:       | |   `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT:       | |     |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT:       | |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT:       | |     | | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |     | | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:       | |     | | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |     | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT:       | |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT:       | |     | | | | `-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |     | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:       | |     | | | |     `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:       | |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count' 'int'
// CHECK-NEXT:       | |     | | `-<<<NULL>>>
// CHECK-NEXT:       | |     | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |     | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:       | |     | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:       | |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count' 'int'
// CHECK-NEXT:       | |     |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |     | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:       | |     |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       | |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:       | |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count' 'int'
// CHECK-NEXT:       | `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT:       |   `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT:       |     `-BinaryOperator {{.+}} 'char *__bidi_indexable' '+'
// CHECK-NEXT:       |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT:       |       | |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT:       |       | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT:       |       | | | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       |       | | | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:       |       | | | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       |       | | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT:       |       | | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT:       |       | | | | | `-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       |       | | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:       |       | | | | |     `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       |       | | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       |       | | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:       |       | | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count' 'int'
// CHECK-NEXT:       |       | | | `-<<<NULL>>>
// CHECK-NEXT:       |       | | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:       |       | | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       |       | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       |       | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:       |       | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count' 'int'
// CHECK-NEXT:       |       | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       |       | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:       |       | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:       |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count' 'int'
// CHECK-NEXT:       |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:       |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count' 'int'
// CHECK-NEXT:       |-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT:       | `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT:       |   `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT:       |     |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT:       |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT:       |     | | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       |     | | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:       |     | | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       |     | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT:       |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT:       |     | | | | `-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       |     | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:       |     | | | |     `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:       |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count' 'int'
// CHECK-NEXT:       |     | | `-<<<NULL>>>
// CHECK-NEXT:       |     | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       |     | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:       |     | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:       |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count' 'int'
// CHECK-NEXT:       |     |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       |     | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:       |     |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:       |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:       |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count' 'int'
// CHECK-NEXT:       `-OpaqueValueExpr {{.+}} 'const char *__bidi_indexable'
// CHECK-NEXT:         `-ImplicitCastExpr {{.+}} 'const char *__bidi_indexable' <NoOp>
// CHECK-NEXT:           `-BinaryOperator {{.+}} 'char *__bidi_indexable' '+'
// CHECK-NEXT:             |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT:             | |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT:             | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT:             | | | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:             | | | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:             | | | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:             | | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT:             | | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT:             | | | | | `-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:             | | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:             | | | | |     `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:             | | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:             | | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:             | | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count' 'int'
// CHECK-NEXT:             | | | `-<<<NULL>>>
// CHECK-NEXT:             | | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:             | | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:             | | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:             | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:             | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:             | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count' 'int'
// CHECK-NEXT:             | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:             | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT:             | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ParmVar {{.+}} 'start' 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT:             | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:             |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:             |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count' 'int'
// CHECK-NEXT:             `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:               `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count' 'int'
void pass_counted_by(char* __counted_by(count) start, int count) {
  ended_by(start, start + count);
}
