// FileCheck lines automatically generated using make-ast-dump-check-v2.py

// RUN: %clang_cc1 -ast-dump -fbounds-safety -fbounds-safety-bringup-missing-checks=compound_literal_init -Wno-bounds-attributes-init-list-side-effect -verify %s | FileCheck %s
// RUN: %clang_cc1 -x objective-c -fexperimental-bounds-safety-objc -ast-dump -fbounds-safety -fbounds-safety-bringup-missing-checks=compound_literal_init -Wno-bounds-attributes-init-list-side-effect -verify %s | FileCheck %s

#include <ptrcheck.h>

// expected-no-diagnostics
struct sb {
  int count;
  char* __sized_by(count) buf;
};
// CHECK-LABEL:|-FunctionDecl {{.+}} used consume_sb 'void (struct sb)'
// CHECK-NEXT: | `-ParmVarDecl {{.+}} 'struct sb'
void consume_sb(struct sb);
// CHECK-LABEL:|-FunctionDecl {{.+}} used consume_sb_arr 'void (struct sb (*__single)[])'
// CHECK-NEXT: | `-ParmVarDecl {{.+}} arr 'struct sb (*__single)[]'
void consume_sb_arr(struct sb (*arr)[]);

struct nested_sb {
  struct sb nested;
  int other;
};

struct nested_and_outer_sb {
  struct sb nested;
  int other;
  int count;
  char* __sized_by(count) buf;
};

// CHECK-LABEL:|-FunctionDecl {{.+}} used get_int 'int (void)'
int get_int(void);

struct sb_with_other_data {
  int count;
  char* __sized_by(count) buf;
  int other;
};
struct no_attr_with_other_data {
  int count;
  char* buf;
  int other;
};
_Static_assert(sizeof(struct sb_with_other_data) == 
               sizeof(struct no_attr_with_other_data), "size mismatch");
// CHECK-LABEL:|-FunctionDecl {{.+}} consume_sb_with_other_data_arr 'void (struct sb_with_other_data (*__single)[])'
// CHECK-NEXT: | `-ParmVarDecl {{.+}} arr 'struct sb_with_other_data (*__single)[]'
void consume_sb_with_other_data_arr(struct sb_with_other_data (*arr)[]);

union TransparentUnion {
  struct sb_with_other_data sb;
  struct no_attr_with_other_data no_sb;
} __attribute__((__transparent_union__));

// CHECK-LABEL:|-FunctionDecl {{.+}} used receive_transparent_union 'void (union TransparentUnion)'
// CHECK-NEXT: | `-ParmVarDecl {{.+}} 'union TransparentUnion'
void receive_transparent_union(union TransparentUnion);

// NOTE: We currently don't test globals because those don't generate bounds
// checks. We currently rely on Sema checks to prevent all invalid externally
// counted pointers. This only works because global initializers must be
// constant evaluable.

// ============================================================================= 
// Tests with __bidi_indexable source ptr
//
// These are more readable due to the lack of BoundsSafetyPromotionExpr
// ============================================================================= 

// CHECK-LABEL:|-FunctionDecl {{.+}} assign_via_ptr 'void (struct sb *__single, int, char *__bidi_indexable)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'struct sb *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__bidi_indexable'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct sb' '='
// CHECK-NEXT: |     |-UnaryOperator {{.+}} 'struct sb' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct sb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct sb' lvalue
// CHECK-NEXT: |         `-BoundsCheckExpr {{.+}} 'struct sb' 'new_ptr <= __builtin_get_pointer_upper_bound(new_ptr) && __builtin_get_pointer_lower_bound(new_ptr) <= new_ptr && new_count <= (char *)__builtin_get_pointer_upper_bound(new_ptr) - (char *__bidi_indexable)new_ptr && 0 <= new_count'
// CHECK-NEXT: |           |-InitListExpr {{.+}} 'struct sb'
// CHECK-NEXT: |           | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | | | | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | | |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | | | |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           | | | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |           | | |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | | |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           | |   |-GetBoundExpr {{.+}} 'char *' lower
// CHECK-NEXT: |           | |   | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |   |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | |   |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           | |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | |     `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |       `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | |         `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: |           |   | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: |           |   |   |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK-NEXT: |           |   |   | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |           |   |   |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |   |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |   |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |           |   |       `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |         `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           |   |           `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           |   `-BinaryOperator {{.+}} <<invalid sloc>, line:{{.+}}> 'int' '<='
// CHECK-NEXT: |           |     |-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |           |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |             `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |               `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
void assign_via_ptr(struct sb* ptr, int new_count,
                    char* __bidi_indexable new_ptr) {
  *ptr = (struct sb) {
    .count = new_count,
    .buf = new_ptr
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} assign_operator 'void (int, char *__bidi_indexable)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__bidi_indexable'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   |-DeclStmt {{.+}}
// CHECK-NEXT: |   | `-VarDecl {{.+}} used new 'struct sb'
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct sb' '='
// CHECK-NEXT: |     |-DeclRefExpr {{.+}} 'struct sb' lvalue Var {{.+}} 'new' 'struct sb'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct sb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct sb' lvalue
// CHECK-NEXT: |         `-BoundsCheckExpr {{.+}} 'struct sb' 'new_ptr <= __builtin_get_pointer_upper_bound(new_ptr) && __builtin_get_pointer_lower_bound(new_ptr) <= new_ptr && new_count <= (char *)__builtin_get_pointer_upper_bound(new_ptr) - (char *__bidi_indexable)new_ptr && 0 <= new_count'
// CHECK-NEXT: |           |-InitListExpr {{.+}} 'struct sb'
// CHECK-NEXT: |           | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | | | | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | | |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | | | |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           | | | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |           | | |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | | |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           | |   |-GetBoundExpr {{.+}} 'char *' lower
// CHECK-NEXT: |           | |   | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |   |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | |   |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           | |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | |     `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |       `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | |         `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: |           |   | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: |           |   |   |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK-NEXT: |           |   |   | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |           |   |   |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |   |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |   |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |           |   |       `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |         `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           |   |           `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           |   `-BinaryOperator {{.+}} <<invalid sloc>, line:{{.+}}> 'int' '<='
// CHECK-NEXT: |           |     |-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |           |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |             `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |               `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
void assign_operator(int new_count, char* __bidi_indexable new_ptr) {
  struct sb new;
  new = (struct sb) {
    .count = new_count,
    .buf = new_ptr
  };
}


// CHECK-LABEL:|-FunctionDecl {{.+}} local_var_init 'void (int, char *__bidi_indexable)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__bidi_indexable'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-DeclStmt {{.+}}
// CHECK-NEXT: |     `-VarDecl {{.+}} new 'struct sb' cinit
// CHECK-NEXT: |       `-ImplicitCastExpr {{.+}} 'struct sb' <LValueToRValue>
// CHECK-NEXT: |         `-CompoundLiteralExpr {{.+}} 'struct sb' lvalue
// CHECK-NEXT: |           `-BoundsCheckExpr {{.+}} 'struct sb' 'new_ptr <= __builtin_get_pointer_upper_bound(new_ptr) && __builtin_get_pointer_lower_bound(new_ptr) <= new_ptr && new_count <= (char *)__builtin_get_pointer_upper_bound(new_ptr) - (char *__bidi_indexable)new_ptr && 0 <= new_count'
// CHECK-NEXT: |             |-InitListExpr {{.+}} 'struct sb'
// CHECK-NEXT: |             | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |             | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |             | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |             | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |             |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |             |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |             |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |             |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |             | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |             | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |             | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |             | | | | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |             | | | |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |             | | | |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |             | | | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |             | | |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |             | | |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |             | | |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |             | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |             | |   |-GetBoundExpr {{.+}} 'char *' lower
// CHECK-NEXT: |             | |   | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |             | |   |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |             | |   |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |             | |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |             | |     `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |             | |       `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |             | |         `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |             | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |             |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |             |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: |             |   | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |             |   | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |             |   | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |             |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: |             |   |   |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK-NEXT: |             |   |   | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |             |   |   |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |             |   |   |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |             |   |   |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |             |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |             |   |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |             |   |       `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |             |   |         `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |             |   |           `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |             |   `-BinaryOperator {{.+}} <<invalid sloc>, line:{{.+}}> 'int' '<='
// CHECK-NEXT: |             |     |-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |             |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |             |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |             |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |             |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |             | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |             |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |             `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |               `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |                 `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
void local_var_init(int new_count, char* __bidi_indexable new_ptr) {
  struct sb new = (struct sb) {
    .count = new_count,
    .buf = new_ptr
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} call_arg 'void (int, char *__bidi_indexable)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__bidi_indexable'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-CallExpr {{.+}} 'void'
// CHECK-NEXT: |     |-ImplicitCastExpr {{.+}} 'void (*__single)(struct sb)' <FunctionToPointerDecay>
// CHECK-NEXT: |     | `-DeclRefExpr {{.+}} 'void (struct sb)' Function {{.+}} 'consume_sb' 'void (struct sb)'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct sb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct sb' lvalue
// CHECK-NEXT: |         `-BoundsCheckExpr {{.+}} 'struct sb' 'new_ptr <= __builtin_get_pointer_upper_bound(new_ptr) && __builtin_get_pointer_lower_bound(new_ptr) <= new_ptr && new_count <= (char *)__builtin_get_pointer_upper_bound(new_ptr) - (char *__bidi_indexable)new_ptr && 0 <= new_count'
// CHECK-NEXT: |           |-InitListExpr {{.+}} 'struct sb'
// CHECK-NEXT: |           | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | | | | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | | |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | | | |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           | | | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |           | | |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | | |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           | |   |-GetBoundExpr {{.+}} 'char *' lower
// CHECK-NEXT: |           | |   | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |   |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | |   |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           | |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | |     `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |       `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | |         `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: |           |   | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: |           |   |   |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK-NEXT: |           |   |   | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |           |   |   |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |   |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |   |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |           |   |       `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |         `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           |   |           `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           |   `-BinaryOperator {{.+}} <<invalid sloc>, line:{{.+}}> 'int' '<='
// CHECK-NEXT: |           |     |-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |           |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |             `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |               `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
void call_arg(int new_count, char* __bidi_indexable new_ptr) {
  consume_sb((struct sb) {
    .count = new_count,
    .buf = new_ptr
  });
}

// CHECK-LABEL:|-FunctionDecl {{.+}} return_sb 'struct sb (int, char *__bidi_indexable)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__bidi_indexable'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-ReturnStmt {{.+}}
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct sb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct sb' lvalue
// CHECK-NEXT: |         `-BoundsCheckExpr {{.+}} 'struct sb' 'new_ptr <= __builtin_get_pointer_upper_bound(new_ptr) && __builtin_get_pointer_lower_bound(new_ptr) <= new_ptr && new_count <= (char *)__builtin_get_pointer_upper_bound(new_ptr) - (char *__bidi_indexable)new_ptr && 0 <= new_count'
// CHECK-NEXT: |           |-InitListExpr {{.+}} 'struct sb'
// CHECK-NEXT: |           | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | | | | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | | |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | | | |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           | | | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |           | | |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | | |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           | |   |-GetBoundExpr {{.+}} 'char *' lower
// CHECK-NEXT: |           | |   | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |   |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | |   |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           | |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | |     `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |       `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | |         `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: |           |   | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: |           |   |   |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK-NEXT: |           |   |   | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |           |   |   |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |   |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |   |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |           |   |       `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |         `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           |   |           `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           |   `-BinaryOperator {{.+}} <<invalid sloc>, line:{{.+}}> 'int' '<='
// CHECK-NEXT: |           |     |-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |           |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |             `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |               `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
struct sb return_sb(int new_count, char* __bidi_indexable new_ptr) {
  return (struct sb) {
    .count = new_count,
    .buf = new_ptr
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} construct_not_used 'void (int, char *__bidi_indexable)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__bidi_indexable'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-CStyleCastExpr {{.+}} 'void' <ToVoid>
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct sb' <LValueToRValue> part_of_explicit_cast
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct sb' lvalue
// CHECK-NEXT: |         `-BoundsCheckExpr {{.+}} 'struct sb' 'new_ptr <= __builtin_get_pointer_upper_bound(new_ptr) && __builtin_get_pointer_lower_bound(new_ptr) <= new_ptr && new_count <= (char *)__builtin_get_pointer_upper_bound(new_ptr) - (char *__bidi_indexable)new_ptr && 0 <= new_count'
// CHECK-NEXT: |           |-InitListExpr {{.+}} 'struct sb'
// CHECK-NEXT: |           | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | | | | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | | |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | | | |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           | | | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |           | | |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | | |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           | |   |-GetBoundExpr {{.+}} 'char *' lower
// CHECK-NEXT: |           | |   | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |   |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | |   |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           | |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | |     `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |       `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | |         `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: |           |   | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: |           |   |   |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK-NEXT: |           |   |   | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |           |   |   |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |   |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |   |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |           |   |       `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |         `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           |   |           `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           |   `-BinaryOperator {{.+}} <<invalid sloc>, line:{{.+}}> 'int' '<='
// CHECK-NEXT: |           |     |-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |           |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |             `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |               `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
void construct_not_used(int new_count, char* __bidi_indexable new_ptr) {
  (void)(struct sb) {
    .count = new_count,
    .buf = new_ptr
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} assign_via_ptr_nullptr 'void (struct sb *__single, int)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'struct sb *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct sb' '='
// CHECK-NEXT: |     |-UnaryOperator {{.+}} 'struct sb' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct sb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct sb' lvalue
// CHECK-NEXT: |         `-BoundsCheckExpr {{.+}} 'struct sb' 'new_count == 0'
// CHECK-NEXT: |           |-InitListExpr {{.+}} 'struct sb'
// CHECK-NEXT: |           | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count)':'char *__single'
// CHECK-NEXT: |           |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <NullToPointer>
// CHECK-NEXT: |           |     `-IntegerLiteral {{.+}} 'int' 0
// CHECK-NEXT: |           |-BinaryOperator {{.+}} <line:{{.+}}, <invalid sloc>> 'int' '=='
// CHECK-NEXT: |           | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | `-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |           |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count)':'char *__single'
// CHECK-NEXT: |             `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <NullToPointer>
// CHECK-NEXT: |               `-IntegerLiteral {{.+}} 'int' 0
void assign_via_ptr_nullptr(struct sb* ptr, int new_count) {
  *ptr = (struct sb) {
    .count = new_count,
    .buf = 0x0
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} assign_via_ptr_nested 'void (struct nested_sb *__single, char *__bidi_indexable, int)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'struct nested_sb *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__bidi_indexable'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct nested_sb' '='
// CHECK-NEXT: |     |-UnaryOperator {{.+}} 'struct nested_sb' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'struct nested_sb *__single' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.+}} 'struct nested_sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct nested_sb *__single'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct nested_sb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct nested_sb' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct nested_sb'
// CHECK-NEXT: |           |-BoundsCheckExpr {{.+}} 'struct sb' 'new_ptr <= __builtin_get_pointer_upper_bound(new_ptr) && __builtin_get_pointer_lower_bound(new_ptr) <= new_ptr && new_count <= (char *)__builtin_get_pointer_upper_bound(new_ptr) - (char *__bidi_indexable)new_ptr && 0 <= new_count'
// CHECK-NEXT: |           | |-InitListExpr {{.+}} 'struct sb'
// CHECK-NEXT: |           | | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           | | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           | | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           | | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | | | | | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | | | |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | | | | |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           | | | | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |           | | | |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | | |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | | | |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           | | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           | | |   |-GetBoundExpr {{.+}} 'char *' lower
// CHECK-NEXT: |           | | |   | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | |   |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | | |   |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           | | |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | | |     `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | |       `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | | |         `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           | |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           | |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: |           | |   | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |   | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: |           | |   |   |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK-NEXT: |           | |   |   | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |           | |   |   |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |   |   |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | |   |   |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           | |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | |   |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |           | |   |       `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |   |         `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | |   |           `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           | |   `-BinaryOperator {{.+}} <<invalid sloc>, col:{{.+}}> 'int' '<='
// CHECK-NEXT: |           | |     |-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |           | |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           `-IntegerLiteral {{.+}} 'int' 0
void assign_via_ptr_nested(struct nested_sb* ptr,
                           char* __bidi_indexable new_ptr,
                           int new_count) {
  *ptr = (struct nested_sb) {
    .nested = {.buf = new_ptr, .count = new_count },
    .other = 0x0
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} assign_via_ptr_nested_v2 'void (struct nested_sb *__single, char *__bidi_indexable, int)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'struct nested_sb *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__bidi_indexable'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct nested_sb' '='
// CHECK-NEXT: |     |-UnaryOperator {{.+}} 'struct nested_sb' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'struct nested_sb *__single' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.+}} 'struct nested_sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct nested_sb *__single'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct nested_sb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct nested_sb' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct nested_sb'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'struct sb' <LValueToRValue>
// CHECK-NEXT: |           | `-CompoundLiteralExpr {{.+}} 'struct sb' lvalue
// CHECK-NEXT: |           |   `-BoundsCheckExpr {{.+}} 'struct sb' 'new_ptr <= __builtin_get_pointer_upper_bound(new_ptr) && __builtin_get_pointer_lower_bound(new_ptr) <= new_ptr && new_count <= (char *)__builtin_get_pointer_upper_bound(new_ptr) - (char *__bidi_indexable)new_ptr && 0 <= new_count'
// CHECK-NEXT: |           |     |-InitListExpr {{.+}} 'struct sb'
// CHECK-NEXT: |           |     | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |     | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |     | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |     | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |     |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |     |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           |     |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           |     |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           |     | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           |     | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |     | | | | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |     | | | |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           |     | | | |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           |     | | | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |           |     | | |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |     | | |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           |     | | |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           |     | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           |     | |   |-GetBoundExpr {{.+}} 'char *' lower
// CHECK-NEXT: |           |     | |   | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |     | |   |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           |     | |   |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           |     | |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |     | |     `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |     | |       `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           |     | |         `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           |     | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           |     |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           |     |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: |           |     |   | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |     |   | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |     |   | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |     |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: |           |     |   |   |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK-NEXT: |           |     |   |   | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |           |     |   |   |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |     |   |   |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           |     |   |   |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           |     |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |     |   |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |           |     |   |       `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |     |   |         `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           |     |   |           `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           |     |   `-BinaryOperator {{.+}} <<invalid sloc>, col:{{.+}}> 'int' '<='
// CHECK-NEXT: |           |     |     |-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |           |     |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |     |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |     |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |     |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |     | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |     |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |     `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |       `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           |         `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           `-IntegerLiteral {{.+}} 'int' 0
void assign_via_ptr_nested_v2(struct nested_sb* ptr,
                              char* __bidi_indexable new_ptr,
                              int new_count) {
  *ptr = (struct nested_sb) {
    .nested = (struct sb){.buf = new_ptr, .count = new_count },
    .other = 0x0
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} assign_via_ptr_nested_v3 'void (struct nested_and_outer_sb *__single, char *__bidi_indexable, int)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'struct nested_and_outer_sb *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__bidi_indexable'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct nested_and_outer_sb' '='
// CHECK-NEXT: |     |-UnaryOperator {{.+}} 'struct nested_and_outer_sb' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'struct nested_and_outer_sb *__single' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.+}} 'struct nested_and_outer_sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct nested_and_outer_sb *__single'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct nested_and_outer_sb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct nested_and_outer_sb' lvalue
// CHECK-NEXT: |         `-BoundsCheckExpr {{.+}} 'struct nested_and_outer_sb' 'new_ptr <= __builtin_get_pointer_upper_bound(new_ptr) && __builtin_get_pointer_lower_bound(new_ptr) <= new_ptr && new_count <= (char *)__builtin_get_pointer_upper_bound(new_ptr) - (char *__bidi_indexable)new_ptr && 0 <= new_count'
// CHECK-NEXT: |           |-InitListExpr {{.+}} 'struct nested_and_outer_sb'
// CHECK-NEXT: |           | |-ImplicitCastExpr {{.+}} 'struct sb' <LValueToRValue>
// CHECK-NEXT: |           | | `-CompoundLiteralExpr {{.+}} 'struct sb' lvalue
// CHECK-NEXT: |           | |   `-BoundsCheckExpr {{.+}} 'struct sb' 'new_ptr <= __builtin_get_pointer_upper_bound(new_ptr) && __builtin_get_pointer_lower_bound(new_ptr) <= new_ptr && new_count <= (char *)__builtin_get_pointer_upper_bound(new_ptr) - (char *__bidi_indexable)new_ptr && 0 <= new_count'
// CHECK-NEXT: |           | |     |-InitListExpr {{.+}} 'struct sb'
// CHECK-NEXT: |           | |     | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |     | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |     | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |     | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | |     |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |     |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | |     |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           | |     |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           | |     | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           | |     | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           | |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | |     | | | | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |     | | | |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | |     | | | |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           | |     | | | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |           | |     | | |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |     | | |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | |     | | |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           | |     | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           | |     | |   |-GetBoundExpr {{.+}} 'char *' lower
// CHECK-NEXT: |           | |     | |   | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |     | |   |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | |     | |   |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           | |     | |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | |     | |     `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |     | |       `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | |     | |         `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           | |     | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           | |     |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           | |     |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: |           | |     |   | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |     |   | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |     |   | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |     |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: |           | |     |   |   |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK-NEXT: |           | |     |   |   | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |           | |     |   |   |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |     |   |   |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | |     |   |   |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           | |     |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | |     |   |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |           | |     |   |       `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |     |   |         `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | |     |   |           `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           | |     |   `-BinaryOperator {{.+}} <<invalid sloc>, col:{{.+}}> 'int' '<='
// CHECK-NEXT: |           | |     |     |-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |           | |     |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |     |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |     |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |     |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |     | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |     |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |     `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |       `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | |         `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           | |-IntegerLiteral {{.+}} 'int' 0
// CHECK-NEXT: |           | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | | | | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | | |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | | | |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           | | | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |           | | |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | | |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           | |   |-GetBoundExpr {{.+}} 'char *' lower
// CHECK-NEXT: |           | |   | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |   |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | |   |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           | |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | |     `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |       `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | |         `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: |           |   | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: |           |   |   |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK-NEXT: |           |   |   | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |           |   |   |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |   |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |   |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |           |   |       `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |         `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           |   |           `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           |   `-BinaryOperator {{.+}} <<invalid sloc>, line:{{.+}}> 'int' '<='
// CHECK-NEXT: |           |     |-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |           |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |             `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |               `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
void assign_via_ptr_nested_v3(struct nested_and_outer_sb* ptr,
                              char* __bidi_indexable new_ptr,
                              int new_count) {
  *ptr = (struct nested_and_outer_sb) {
    .nested = (struct sb){.buf = new_ptr, .count = new_count },
    .other = 0x0,
    .buf = new_ptr,
    .count = new_count
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} array_of_struct_init 'void (char *__bidi_indexable, int)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__bidi_indexable'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   |-DeclStmt {{.+}}
// CHECK-NEXT: |   | `-VarDecl {{.+}} used arr 'struct sb[2]' cinit
// CHECK-NEXT: |   |   `-CompoundLiteralExpr {{.+}} 'struct sb[2]'
// CHECK-NEXT: |   |     `-InitListExpr {{.+}} 'struct sb[2]'
// CHECK-NEXT: |   |       |-BoundsCheckExpr {{.+}} 'struct sb' 'new_ptr <= __builtin_get_pointer_upper_bound(new_ptr) && __builtin_get_pointer_lower_bound(new_ptr) <= new_ptr && new_count <= (char *)__builtin_get_pointer_upper_bound(new_ptr) - (char *__bidi_indexable)new_ptr && 0 <= new_count'
// CHECK-NEXT: |   |       | |-InitListExpr {{.+}} 'struct sb'
// CHECK-NEXT: |   |       | | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |       | | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |       | | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |   |       | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |       | |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |       | |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |       | |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |   |       | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |       | | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |       | | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |       | | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |       | | | | | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |       | | | | |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |       | | | | |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |   |       | | | | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |   |       | | | |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |       | | | |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |       | | | |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |   |       | | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |       | | |   |-GetBoundExpr {{.+}} 'char *' lower
// CHECK-NEXT: |   |       | | |   | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |       | | |   |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |       | | |   |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |   |       | | |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |       | | |     `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |       | | |       `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |       | | |         `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |   |       | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |       | |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |       | |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: |   |       | |   | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |       | |   | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |       | |   | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |   |       | |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: |   |       | |   |   |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK-NEXT: |   |       | |   |   | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |   |       | |   |   |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |       | |   |   |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |       | |   |   |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |   |       | |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |       | |   |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |   |       | |   |       `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |       | |   |         `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |       | |   |           `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |   |       | |   `-BinaryOperator {{.+}} <<invalid sloc>, col:{{.+}}> 'int' '<='
// CHECK-NEXT: |   |       | |     |-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |   |       | |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |       | |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |       | |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |   |       | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |       | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |       | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |   |       | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |       |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |       |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |   |       `-BoundsCheckExpr {{.+}} 'struct sb' '0 == 0'
// CHECK-NEXT: |   |         |-InitListExpr {{.+}} 'struct sb'
// CHECK-NEXT: |   |         | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | | `-IntegerLiteral {{.+}} 'int' 0
// CHECK-NEXT: |   |         | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count)':'char *__single'
// CHECK-NEXT: |   |         |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <NullToPointer>
// CHECK-NEXT: |   |         |     `-IntegerLiteral {{.+}} 'int' 0
// CHECK-NEXT: |   |         |-BinaryOperator {{.+}} <col:{{.+}}, <invalid sloc>> 'int' '=='
// CHECK-NEXT: |   |         | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | | `-IntegerLiteral {{.+}} 'int' 0
// CHECK-NEXT: |   |         | `-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |   |         |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | `-IntegerLiteral {{.+}} 'int' 0
// CHECK-NEXT: |   |         `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count)':'char *__single'
// CHECK-NEXT: |   |           `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <NullToPointer>
// CHECK-NEXT: |   |             `-IntegerLiteral {{.+}} 'int' 0
// CHECK-NEXT: |   `-CallExpr {{.+}} 'void'
// CHECK-NEXT: |     |-ImplicitCastExpr {{.+}} 'void (*__single)(struct sb (*__single)[])' <FunctionToPointerDecay>
// CHECK-NEXT: |     | `-DeclRefExpr {{.+}} 'void (struct sb (*__single)[])' Function {{.+}} 'consume_sb_arr' 'void (struct sb (*__single)[])'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct sb (*__single)[]' <BoundsSafetyPointerCast>
// CHECK-NEXT: |       `-ImplicitCastExpr {{.+}} 'struct sb (*__bidi_indexable)[]' <BitCast>
// CHECK-NEXT: |         `-UnaryOperator {{.+}} 'struct sb (*__bidi_indexable)[2]' prefix '&' cannot overflow
// CHECK-NEXT: |           `-DeclRefExpr {{.+}} 'struct sb[2]' lvalue Var {{.+}} 'arr' 'struct sb[2]'
void array_of_struct_init(char* __bidi_indexable new_ptr,
                          int new_count) {
  struct sb arr[] = (struct sb[]) {
    {.count = new_count, .buf = new_ptr},
    // Note the type of `.buf`'s assignment here looks odd in the AST (child of
    // BoundsCheckExpr that should be materialized) because the compound literal
    // is used as the base. E.g.:
    // 'char *__single __sized_by((struct sb[2]){{.count = new_count, .buf = new_ptr}, {.count = 0, .buf = 0}}[1UL].count)':'char *__single'
    //
    // We could consider constant folding the expression
    // to `__sized_by(0)` but this doesn't seem to be necessary.
    {.count = 0x0, .buf = 0x0}
  };
  consume_sb_arr(&arr);
}


// CHECK-LABEL:|-FunctionDecl {{.+}} assign_via_ptr_other_data_side_effect 'void (struct sb_with_other_data *__single, int, char *__bidi_indexable)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'struct sb_with_other_data *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__bidi_indexable'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct sb_with_other_data' '='
// CHECK-NEXT: |     |-UnaryOperator {{.+}} 'struct sb_with_other_data' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'struct sb_with_other_data *__single' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.+}} 'struct sb_with_other_data *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb_with_other_data *__single'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct sb_with_other_data' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct sb_with_other_data' lvalue
// CHECK-NEXT: |         `-BoundsCheckExpr {{.+}} 'struct sb_with_other_data' 'new_ptr <= __builtin_get_pointer_upper_bound(new_ptr) && __builtin_get_pointer_lower_bound(new_ptr) <= new_ptr && new_count <= (char *)__builtin_get_pointer_upper_bound(new_ptr) - (char *__bidi_indexable)new_ptr && 0 <= new_count'
// CHECK-NEXT: |           |-InitListExpr {{.+}} 'struct sb_with_other_data'
// CHECK-NEXT: |           | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           | `-CallExpr {{.+}} 'int'
// CHECK-NEXT: |           |   `-ImplicitCastExpr {{.+}} 'int (*__single)(void)' <FunctionToPointerDecay>
// CHECK-NEXT: |           |     `-DeclRefExpr {{.+}} 'int (void)' Function {{.+}} 'get_int' 'int (void)'
// CHECK-NEXT: |           |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | | | | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | | |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | | | |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           | | | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |           | | |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | | |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           | |   |-GetBoundExpr {{.+}} 'char *' lower
// CHECK-NEXT: |           | |   | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |   |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | |   |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           | |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | |     `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |       `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | |         `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: |           |   | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: |           |   |   |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK-NEXT: |           |   |   | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |           |   |   |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |   |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |   |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |           |   |       `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |         `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           |   |           `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           |   `-BinaryOperator {{.+}} <<invalid sloc>, line:{{.+}}> 'int' '<='
// CHECK-NEXT: |           |     |-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |           |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |             `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |               `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
void assign_via_ptr_other_data_side_effect(struct sb_with_other_data* ptr,
                                           int new_count,
                                           char* __bidi_indexable new_ptr) {
  *ptr = (struct sb_with_other_data) {
    .count = new_count,
    .buf = new_ptr,
    // Side effect. Generates a warning but it is suppressed for this test
    .other = get_int()
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} assign_via_ptr_other_data_side_effect_zero_ptr 'void (struct sb_with_other_data *__single, int, char *__bidi_indexable)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'struct sb_with_other_data *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} new_ptr 'char *__bidi_indexable'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct sb_with_other_data' '='
// CHECK-NEXT: |     |-UnaryOperator {{.+}} 'struct sb_with_other_data' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'struct sb_with_other_data *__single' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.+}} 'struct sb_with_other_data *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb_with_other_data *__single'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct sb_with_other_data' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct sb_with_other_data' lvalue
// CHECK-NEXT: |         `-BoundsCheckExpr {{.+}} 'struct sb_with_other_data' 'new_count == 0'
// CHECK-NEXT: |           |-InitListExpr {{.+}} 'struct sb_with_other_data'
// CHECK-NEXT: |           | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count)':'char *__single'
// CHECK-NEXT: |           | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <NullToPointer>
// CHECK-NEXT: |           | |   `-IntegerLiteral {{.+}} 'int' 0
// CHECK-NEXT: |           | `-CallExpr {{.+}} 'int'
// CHECK-NEXT: |           |   `-ImplicitCastExpr {{.+}} 'int (*__single)(void)' <FunctionToPointerDecay>
// CHECK-NEXT: |           |     `-DeclRefExpr {{.+}} 'int (void)' Function {{.+}} 'get_int' 'int (void)'
// CHECK-NEXT: |           |-BinaryOperator {{.+}} <line:{{.+}}, <invalid sloc>> 'int' '=='
// CHECK-NEXT: |           | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | `-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |           |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count)':'char *__single'
// CHECK-NEXT: |             `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <NullToPointer>
// CHECK-NEXT: |               `-IntegerLiteral {{.+}} 'int' 0
void assign_via_ptr_other_data_side_effect_zero_ptr(struct sb_with_other_data* ptr,
                                           int new_count,
                                           char* __bidi_indexable new_ptr) {
  *ptr = (struct sb_with_other_data) {
    .count = new_count,
    .buf = 0x0,
    // Side effect. Generates a warning but it is suppressed for this test
    .other = get_int()
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} call_arg_transparent_union 'void (int, char *__bidi_indexable)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__bidi_indexable'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-CallExpr {{.+}} 'void'
// CHECK-NEXT: |     |-ImplicitCastExpr {{.+}} 'void (*__single)(union TransparentUnion)' <FunctionToPointerDecay>
// CHECK-NEXT: |     | `-DeclRefExpr {{.+}} 'void (union TransparentUnion)' Function {{.+}} 'receive_transparent_union' 'void (union TransparentUnion)'
// CHECK-NEXT: |     `-CompoundLiteralExpr {{.+}} 'union TransparentUnion'
// CHECK-NEXT: |       `-InitListExpr {{.+}} 'union TransparentUnion' field Field {{.+}} 'sb' 'struct sb_with_other_data'
// CHECK-NEXT: |         `-ImplicitCastExpr {{.+}} 'struct sb_with_other_data' <LValueToRValue>
// CHECK-NEXT: |           `-CompoundLiteralExpr {{.+}} 'struct sb_with_other_data' lvalue
// CHECK-NEXT: |             `-BoundsCheckExpr {{.+}} 'struct sb_with_other_data' 'new_ptr <= __builtin_get_pointer_upper_bound(new_ptr) && __builtin_get_pointer_lower_bound(new_ptr) <= new_ptr && new_count <= (char *)__builtin_get_pointer_upper_bound(new_ptr) - (char *__bidi_indexable)new_ptr && 0 <= new_count'
// CHECK-NEXT: |               |-InitListExpr {{.+}} 'struct sb_with_other_data'
// CHECK-NEXT: |               | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |               | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |               | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |               | |-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |               | | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |               | |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |               | |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |               | `-IntegerLiteral {{.+}} 'int' 0
// CHECK-NEXT: |               |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |               | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |               | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |               | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |               | | | | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |               | | | |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |               | | | |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |               | | | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |               | | |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |               | | |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |               | | |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |               | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |               | |   |-GetBoundExpr {{.+}} 'char *' lower
// CHECK-NEXT: |               | |   | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |               | |   |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |               | |   |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |               | |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |               | |     `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |               | |       `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |               | |         `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |               | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |               |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |               |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: |               |   | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |               |   | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |               |   | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |               |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: |               |   |   |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK-NEXT: |               |   |   | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |               |   |   |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |               |   |   |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |               |   |   |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |               |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |               |   |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |               |   |       `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |               |   |         `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |               |   |           `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |               |   `-BinaryOperator {{.+}} <<invalid sloc>, line:{{.+}}> 'int' '<='
// CHECK-NEXT: |               |     |-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |               |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |               |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |               |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |               |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |               | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |               |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |               `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |                 `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |                   `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
void call_arg_transparent_union(int new_count,
                                char* __bidi_indexable new_ptr) {
  receive_transparent_union(
    (struct sb_with_other_data) {
      .count = new_count,
      .buf = new_ptr,
      .other = 0x0
    }
  );
}

// CHECK-LABEL:|-FunctionDecl {{.+}} call_arg_transparent_union_untransparently 'void (int, char *__bidi_indexable)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__bidi_indexable'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-CallExpr {{.+}} 'void'
// CHECK-NEXT: |     |-ImplicitCastExpr {{.+}} 'void (*__single)(union TransparentUnion)' <FunctionToPointerDecay>
// CHECK-NEXT: |     | `-DeclRefExpr {{.+}} 'void (union TransparentUnion)' Function {{.+}} 'receive_transparent_union' 'void (union TransparentUnion)'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'union TransparentUnion' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'union TransparentUnion' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'union TransparentUnion' field Field {{.+}} 'sb' 'struct sb_with_other_data'
// CHECK-NEXT: |           `-BoundsCheckExpr {{.+}} 'struct sb_with_other_data' 'new_ptr <= __builtin_get_pointer_upper_bound(new_ptr) && __builtin_get_pointer_lower_bound(new_ptr) <= new_ptr && new_count <= (char *)__builtin_get_pointer_upper_bound(new_ptr) - (char *__bidi_indexable)new_ptr && 0 <= new_count'
// CHECK-NEXT: |             |-InitListExpr {{.+}} 'struct sb_with_other_data'
// CHECK-NEXT: |             | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |             | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |             | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |             | |-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |             | | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |             | |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |             | |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |             | `-IntegerLiteral {{.+}} 'int' 0
// CHECK-NEXT: |             |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |             | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |             | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |             | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |             | | | | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |             | | | |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |             | | | |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |             | | | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |             | | |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |             | | |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |             | | |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |             | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |             | |   |-GetBoundExpr {{.+}} 'char *' lower
// CHECK-NEXT: |             | |   | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |             | |   |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |             | |   |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |             | |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |             | |     `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |             | |       `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |             | |         `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |             | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |             |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |             |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: |             |   | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |             |   | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |             |   | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |             |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: |             |   |   |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK-NEXT: |             |   |   | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |             |   |   |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |             |   |   |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |             |   |   |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |             |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |             |   |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |             |   |       `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |             |   |         `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |             |   |           `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |             |   `-BinaryOperator {{.+}} <<invalid sloc>, line:{{.+}}> 'int' '<='
// CHECK-NEXT: |             |     |-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |             |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |             |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |             |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |             |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |             | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |             |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |             `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |               `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |                 `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
void call_arg_transparent_union_untransparently(int new_count,
                                char* __bidi_indexable new_ptr) {
  receive_transparent_union(
    (union TransparentUnion) {
      .sb = {
        .count = new_count,
        .buf = new_ptr,
        .other = 0x0
      }
    }
  );
}

// ============================================================================= 
// Tests with __sized_by source ptr
//
// These are less readable due to the BoundsSafetyPromotionExpr
// ============================================================================= 
// CHECK-LABEL:|-FunctionDecl {{.+}} assign_via_ptr_from_ptr 'void (struct sb *__single)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'struct sb *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct sb' '='
// CHECK-NEXT: |     |-UnaryOperator {{.+}} 'struct sb' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct sb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct sb' lvalue
// CHECK-NEXT: |         `-BoundsCheckExpr {{.+}} 'struct sb' 'ptr->buf <= __builtin_get_pointer_upper_bound(ptr->buf) && __builtin_get_pointer_lower_bound(ptr->buf) <= ptr->buf && ptr->count <= (char *)__builtin_get_pointer_upper_bound(ptr->buf) - (char *__bidi_indexable)ptr->buf && 0 <= ptr->count'
// CHECK-NEXT: |           |-InitListExpr {{.+}} 'struct sb'
// CHECK-NEXT: |           | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   `-MemberExpr {{.+}} 'int' lvalue ->count {{.+}}
// CHECK-NEXT: |           | |     `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           | |       `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |       | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count)':'char *__single'
// CHECK-NEXT: |           |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |       | | |   `-MemberExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' lvalue ->buf {{.+}}
// CHECK-NEXT: |           |       | | |     `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           |       | | |       `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           |       | | |         `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count)':'char *__single'
// CHECK-NEXT: |           |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |       | | | |     `-MemberExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' lvalue ->buf {{.+}}
// CHECK-NEXT: |           |       | | | |       `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           |       | | | |         `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           |       | | | |           `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |       | | |     `-MemberExpr {{.+}} 'int' lvalue ->count {{.+}}
// CHECK-NEXT: |           |       | | |       `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           |       | | |         `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           |       | | |           `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           |       | | `-<<<NULL>>>
// CHECK-NEXT: |           |       | |-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           |       | | `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           |       | |   `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           |       | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |       | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |       | |   `-MemberExpr {{.+}} 'int' lvalue ->count {{.+}}
// CHECK-NEXT: |           |       | |     `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           |       | |       `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           |       | |         `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           |       | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count)':'char *__single'
// CHECK-NEXT: |           |       |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |       |     `-MemberExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' lvalue ->buf {{.+}}
// CHECK-NEXT: |           |       |       `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           |       |         `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           |       |           `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           |       |-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           |       | `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           |       |   `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           |       |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |       | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |       |   `-MemberExpr {{.+}} 'int' lvalue ->count {{.+}}
// CHECK-NEXT: |           |       |     `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           |       |       `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           |       |         `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           |       `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count)':'char *__single'
// CHECK-NEXT: |           |         `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |           `-MemberExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' lvalue ->buf {{.+}}
// CHECK-NEXT: |           |             `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           |               `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           |                 `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | | | | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | | |   `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           | | | |     |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           | | | |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | | |     | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count)':'char *__single'
// CHECK-NEXT: |           | | | |     | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | |     | | |   `-MemberExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' lvalue ->buf {{.+}}
// CHECK-NEXT: |           | | | |     | | |     `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           | | | |     | | |       `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | |     | | |         `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           | | | |     | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           | | | |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | | | |     | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count)':'char *__single'
// CHECK-NEXT: |           | | | |     | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | |     | | | |     `-MemberExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' lvalue ->buf {{.+}}
// CHECK-NEXT: |           | | | |     | | | |       `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           | | | |     | | | |         `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | |     | | | |           `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           | | | |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | | |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | | |     | | |     `-MemberExpr {{.+}} 'int' lvalue ->count {{.+}}
// CHECK-NEXT: |           | | | |     | | |       `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           | | | |     | | |         `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | |     | | |           `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           | | | |     | | `-<<<NULL>>>
// CHECK-NEXT: |           | | | |     | |-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           | | | |     | | `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | |     | |   `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           | | | |     | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | | |     | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | | |     | |   `-MemberExpr {{.+}} 'int' lvalue ->count {{.+}}
// CHECK-NEXT: |           | | | |     | |     `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           | | | |     | |       `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | |     | |         `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           | | | |     | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count)':'char *__single'
// CHECK-NEXT: |           | | | |     |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | |     |     `-MemberExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' lvalue ->buf {{.+}}
// CHECK-NEXT: |           | | | |     |       `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           | | | |     |         `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | |     |           `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           | | | |     |-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           | | | |     | `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | |     |   `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           | | | |     |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | | |     | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | | |     |   `-MemberExpr {{.+}} 'int' lvalue ->count {{.+}}
// CHECK-NEXT: |           | | | |     |     `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           | | | |     |       `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | |     |         `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           | | | |     `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count)':'char *__single'
// CHECK-NEXT: |           | | | |       `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | |         `-MemberExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' lvalue ->buf {{.+}}
// CHECK-NEXT: |           | | | |           `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           | | | |             `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | |               `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           | | | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |           | | |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           | | |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           | | |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | |       | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count)':'char *__single'
// CHECK-NEXT: |           | | |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |       | | |   `-MemberExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' lvalue ->buf {{.+}}
// CHECK-NEXT: |           | | |       | | |     `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           | | |       | | |       `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |       | | |         `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           | | |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           | | |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | | |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count)':'char *__single'
// CHECK-NEXT: |           | | |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |       | | | |     `-MemberExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' lvalue ->buf {{.+}}
// CHECK-NEXT: |           | | |       | | | |       `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           | | |       | | | |         `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |       | | | |           `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           | | |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | |       | | |     `-MemberExpr {{.+}} 'int' lvalue ->count {{.+}}
// CHECK-NEXT: |           | | |       | | |       `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           | | |       | | |         `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |       | | |           `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           | | |       | | `-<<<NULL>>>
// CHECK-NEXT: |           | | |       | |-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           | | |       | | `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |       | |   `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           | | |       | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | |       | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | |       | |   `-MemberExpr {{.+}} 'int' lvalue ->count {{.+}}
// CHECK-NEXT: |           | | |       | |     `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           | | |       | |       `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |       | |         `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           | | |       | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count)':'char *__single'
// CHECK-NEXT: |           | | |       |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |       |     `-MemberExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' lvalue ->buf {{.+}}
// CHECK-NEXT: |           | | |       |       `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           | | |       |         `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |       |           `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           | | |       |-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           | | |       | `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |       |   `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           | | |       |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | |       | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | |       |   `-MemberExpr {{.+}} 'int' lvalue ->count {{.+}}
// CHECK-NEXT: |           | | |       |     `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           | | |       |       `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |       |         `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           | | |       `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count)':'char *__single'
// CHECK-NEXT: |           | | |         `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |           `-MemberExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' lvalue ->buf {{.+}}
// CHECK-NEXT: |           | | |             `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           | | |               `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |                 `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           | |   |-GetBoundExpr {{.+}} 'char *' lower
// CHECK-NEXT: |           | |   | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |   |   `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           | |   |     |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           | |   |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |   |     | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count)':'char *__single'
// CHECK-NEXT: |           | |   |     | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |     | | |   `-MemberExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' lvalue ->buf {{.+}}
// CHECK-NEXT: |           | |   |     | | |     `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           | |   |     | | |       `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |     | | |         `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           | |   |     | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           | |   |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | |   |     | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count)':'char *__single'
// CHECK-NEXT: |           | |   |     | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |     | | | |     `-MemberExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' lvalue ->buf {{.+}}
// CHECK-NEXT: |           | |   |     | | | |       `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           | |   |     | | | |         `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |     | | | |           `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           | |   |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |   |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   |     | | |     `-MemberExpr {{.+}} 'int' lvalue ->count {{.+}}
// CHECK-NEXT: |           | |   |     | | |       `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           | |   |     | | |         `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |     | | |           `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           | |   |     | | `-<<<NULL>>>
// CHECK-NEXT: |           | |   |     | |-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           | |   |     | | `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |     | |   `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           | |   |     | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |   |     | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   |     | |   `-MemberExpr {{.+}} 'int' lvalue ->count {{.+}}
// CHECK-NEXT: |           | |   |     | |     `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           | |   |     | |       `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |     | |         `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           | |   |     | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count)':'char *__single'
// CHECK-NEXT: |           | |   |     |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |     |     `-MemberExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' lvalue ->buf {{.+}}
// CHECK-NEXT: |           | |   |     |       `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           | |   |     |         `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |     |           `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           | |   |     |-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           | |   |     | `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |     |   `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           | |   |     |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |   |     | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   |     |   `-MemberExpr {{.+}} 'int' lvalue ->count {{.+}}
// CHECK-NEXT: |           | |   |     |     `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           | |   |     |       `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |     |         `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           | |   |     `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count)':'char *__single'
// CHECK-NEXT: |           | |   |       `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |         `-MemberExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' lvalue ->buf {{.+}}
// CHECK-NEXT: |           | |   |           `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           | |   |             `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |               `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           | |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | |     `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |       `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           | |         |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           | |         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |         | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count)':'char *__single'
// CHECK-NEXT: |           | |         | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |         | | |   `-MemberExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' lvalue ->buf {{.+}}
// CHECK-NEXT: |           | |         | | |     `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           | |         | | |       `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           | |         | | |         `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           | |         | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           | |         | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | |         | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count)':'char *__single'
// CHECK-NEXT: |           | |         | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |         | | | |     `-MemberExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' lvalue ->buf {{.+}}
// CHECK-NEXT: |           | |         | | | |       `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           | |         | | | |         `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           | |         | | | |           `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           | |         | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |         | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |         | | |     `-MemberExpr {{.+}} 'int' lvalue ->count {{.+}}
// CHECK-NEXT: |           | |         | | |       `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           | |         | | |         `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           | |         | | |           `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           | |         | | `-<<<NULL>>>
// CHECK-NEXT: |           | |         | |-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           | |         | | `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           | |         | |   `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           | |         | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |         | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |         | |   `-MemberExpr {{.+}} 'int' lvalue ->count {{.+}}
// CHECK-NEXT: |           | |         | |     `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           | |         | |       `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           | |         | |         `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           | |         | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count)':'char *__single'
// CHECK-NEXT: |           | |         |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |         |     `-MemberExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' lvalue ->buf {{.+}}
// CHECK-NEXT: |           | |         |       `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           | |         |         `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           | |         |           `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           | |         |-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           | |         | `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           | |         |   `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           | |         |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |         | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |         |   `-MemberExpr {{.+}} 'int' lvalue ->count {{.+}}
// CHECK-NEXT: |           | |         |     `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           | |         |       `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           | |         |         `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           | |         `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count)':'char *__single'
// CHECK-NEXT: |           | |           `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |             `-MemberExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' lvalue ->buf {{.+}}
// CHECK-NEXT: |           | |               `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           | |                 `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           | |                   `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: |           |   | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   | |     `-MemberExpr {{.+}} 'int' lvalue ->count {{.+}}
// CHECK-NEXT: |           |   | |       `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           |   | |         `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: |           |   |   |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK-NEXT: |           |   |   | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |           |   |   |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |   |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           |   |   |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           |   |   |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |   |       | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count)':'char *__single'
// CHECK-NEXT: |           |   |   |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       | | |   `-MemberExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' lvalue ->buf {{.+}}
// CHECK-NEXT: |           |   |   |       | | |     `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           |   |   |       | | |       `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       | | |         `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           |   |   |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           |   |   |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |   |   |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count)':'char *__single'
// CHECK-NEXT: |           |   |   |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       | | | |     `-MemberExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' lvalue ->buf {{.+}}
// CHECK-NEXT: |           |   |   |       | | | |       `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           |   |   |       | | | |         `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       | | | |           `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           |   |   |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   |   |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       | | |     `-MemberExpr {{.+}} 'int' lvalue ->count {{.+}}
// CHECK-NEXT: |           |   |   |       | | |       `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           |   |   |       | | |         `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       | | |           `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           |   |   |       | | `-<<<NULL>>>
// CHECK-NEXT: |           |   |   |       | |-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           |   |   |       | | `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       | |   `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           |   |   |       | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   |   |       | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       | |   `-MemberExpr {{.+}} 'int' lvalue ->count {{.+}}
// CHECK-NEXT: |           |   |   |       | |     `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           |   |   |       | |       `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       | |         `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           |   |   |       | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count)':'char *__single'
// CHECK-NEXT: |           |   |   |       |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       |     `-MemberExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' lvalue ->buf {{.+}}
// CHECK-NEXT: |           |   |   |       |       `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           |   |   |       |         `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       |           `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           |   |   |       |-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           |   |   |       | `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       |   `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           |   |   |       |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   |   |       | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       |   `-MemberExpr {{.+}} 'int' lvalue ->count {{.+}}
// CHECK-NEXT: |           |   |   |       |     `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           |   |   |       |       `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       |         `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           |   |   |       `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count)':'char *__single'
// CHECK-NEXT: |           |   |   |         `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |   |           `-MemberExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' lvalue ->buf {{.+}}
// CHECK-NEXT: |           |   |   |             `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           |   |   |               `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |   |                 `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |   |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |           |   |       `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |         `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           |   |           |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           |   |           | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |           | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count)':'char *__single'
// CHECK-NEXT: |           |   |           | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |           | | |   `-MemberExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' lvalue ->buf {{.+}}
// CHECK-NEXT: |           |   |           | | |     `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           |   |           | | |       `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |           | | |         `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           |   |           | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           |   |           | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |   |           | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count)':'char *__single'
// CHECK-NEXT: |           |   |           | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |           | | | |     `-MemberExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' lvalue ->buf {{.+}}
// CHECK-NEXT: |           |   |           | | | |       `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           |   |           | | | |         `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |           | | | |           `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           |   |           | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   |           | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   |           | | |     `-MemberExpr {{.+}} 'int' lvalue ->count {{.+}}
// CHECK-NEXT: |           |   |           | | |       `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           |   |           | | |         `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |           | | |           `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           |   |           | | `-<<<NULL>>>
// CHECK-NEXT: |           |   |           | |-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           |   |           | | `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |           | |   `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           |   |           | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   |           | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   |           | |   `-MemberExpr {{.+}} 'int' lvalue ->count {{.+}}
// CHECK-NEXT: |           |   |           | |     `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           |   |           | |       `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |           | |         `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           |   |           | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count)':'char *__single'
// CHECK-NEXT: |           |   |           |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |           |     `-MemberExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' lvalue ->buf {{.+}}
// CHECK-NEXT: |           |   |           |       `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           |   |           |         `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |           |           `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           |   |           |-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           |   |           | `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |           |   `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           |   |           |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   |           | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   |           |   `-MemberExpr {{.+}} 'int' lvalue ->count {{.+}}
// CHECK-NEXT: |           |   |           |     `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           |   |           |       `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |           |         `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           |   |           `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count)':'char *__single'
// CHECK-NEXT: |           |   |             `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |               `-MemberExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' lvalue ->buf {{.+}}
// CHECK-NEXT: |           |   |                 `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |           |   |                   `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |                     `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           |   `-BinaryOperator {{.+}} <<invalid sloc>, line:{{.+}}> 'int' '<='
// CHECK-NEXT: |           |     |-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |           |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |         `-MemberExpr {{.+}} 'int' lvalue ->count {{.+}}
// CHECK-NEXT: |           |           `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           |             `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   `-MemberExpr {{.+}} 'int' lvalue ->count {{.+}}
// CHECK-NEXT: |           |     `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |           |       `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |           `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |             `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |               |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |               | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |               | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count)':'char *__single'
// CHECK-NEXT: |               | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               | | |   `-MemberExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' lvalue ->buf {{.+}}
// CHECK-NEXT: |               | | |     `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |               | | |       `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |               | | |         `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |               | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |               | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |               | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count)':'char *__single'
// CHECK-NEXT: |               | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               | | | |     `-MemberExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' lvalue ->buf {{.+}}
// CHECK-NEXT: |               | | | |       `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |               | | | |         `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |               | | | |           `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |               | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |               | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |               | | |     `-MemberExpr {{.+}} 'int' lvalue ->count {{.+}}
// CHECK-NEXT: |               | | |       `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |               | | |         `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |               | | |           `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |               | | `-<<<NULL>>>
// CHECK-NEXT: |               | |-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |               | | `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |               | |   `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |               | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |               | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |               | |   `-MemberExpr {{.+}} 'int' lvalue ->count {{.+}}
// CHECK-NEXT: |               | |     `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |               | |       `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |               | |         `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |               | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count)':'char *__single'
// CHECK-NEXT: |               |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               |     `-MemberExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' lvalue ->buf {{.+}}
// CHECK-NEXT: |               |       `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |               |         `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |               |           `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |               |-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |               | `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |               |   `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |               |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |               | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |               |   `-MemberExpr {{.+}} 'int' lvalue ->count {{.+}}
// CHECK-NEXT: |               |     `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |               |       `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |               |         `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |               `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count)':'char *__single'
// CHECK-NEXT: |                 `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |                   `-MemberExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' lvalue ->buf {{.+}}
// CHECK-NEXT: |                     `-OpaqueValueExpr {{.+}} 'struct sb *__single'
// CHECK-NEXT: |                       `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |                         `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
void assign_via_ptr_from_ptr(struct sb* ptr) {
  *ptr = (struct sb) {
    .count = ptr->count,
    .buf = ptr->buf
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} assign_via_ptr_from_sb 'void (struct sb *__single, int, char *__single __sized_by(new_count))'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'struct sb *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | | `-DependerDeclsAttr {{.+}} <<invalid sloc>> Implicit {{.+}} 0
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct sb' '='
// CHECK-NEXT: |     |-UnaryOperator {{.+}} 'struct sb' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'struct sb *__single' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.+}} 'struct sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb *__single'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct sb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct sb' lvalue
// CHECK-NEXT: |         `-BoundsCheckExpr {{.+}} 'struct sb' 'new_ptr <= __builtin_get_pointer_upper_bound(new_ptr) && __builtin_get_pointer_lower_bound(new_ptr) <= new_ptr && new_count <= (char *)__builtin_get_pointer_upper_bound(new_ptr) - (char *__bidi_indexable)new_ptr && 0 <= new_count'
// CHECK-NEXT: |           |-InitListExpr {{.+}} 'struct sb'
// CHECK-NEXT: |           | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |       | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |       | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |       | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |       | | `-<<<NULL>>>
// CHECK-NEXT: |           |       | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |       | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |       | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |       |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |       | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |       |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | | | | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | | |   `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           | | | |     |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           | | | |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | | |     | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | |     | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           | | | |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | | | |     | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | |     | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | | |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | | |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | | |     | | `-<<<NULL>>>
// CHECK-NEXT: |           | | | |     | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | |     | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | | |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | | |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | | |     |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | |     |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | | |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | | |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |           | | |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           | | |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           | | |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | |       | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |       | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           | | |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | | |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |       | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | |       | | `-<<<NULL>>>
// CHECK-NEXT: |           | | |       | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |       | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | |       |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |       |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           | |   |-GetBoundExpr {{.+}} 'char *' lower
// CHECK-NEXT: |           | |   | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |   |   `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           | |   |     |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           | |   |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |   |     | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |     | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           | |   |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | |   |     | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |     | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |   |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |   |     | | `-<<<NULL>>>
// CHECK-NEXT: |           | |   |     | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |     | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |   |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |   |     |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |     |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |   |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | |     `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |       `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           | |         |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           | |         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |         | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |         | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           | |         | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | |         | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |         | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |         | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |         | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |         | | `-<<<NULL>>>
// CHECK-NEXT: |           | |         | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |         | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |         |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |         |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |         |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |         |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |           `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |             `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: |           |   | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: |           |   |   |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK-NEXT: |           |   |   | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |           |   |   |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |   |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           |   |   |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           |   |   |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |   |       | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           |   |   |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |   |   |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   |   |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   |   |       | | `-<<<NULL>>>
// CHECK-NEXT: |           |   |   |       | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   |   |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   |   |       |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   |   |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   |   |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |   |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |           |   |       `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |         `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           |   |           |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           |   |           | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |           | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |           | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           |   |           | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |   |           | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |           | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   |           | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   |           | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   |           | | `-<<<NULL>>>
// CHECK-NEXT: |           |   |           | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |           | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   |           |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   |           |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   |           |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |           |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   |             `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   |               `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   `-BinaryOperator {{.+}} <<invalid sloc>, line:{{.+}}> 'int' '<='
// CHECK-NEXT: |           |     |-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |           |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |             `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |               |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |               | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |               | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |               | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |               | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |               | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |               | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |               | | `-<<<NULL>>>
// CHECK-NEXT: |               | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |               |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |               |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |               |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |                 `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |                   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
void assign_via_ptr_from_sb(struct sb* ptr, int new_count,
                            char* __sized_by(new_count) new_ptr) {
  *ptr = (struct sb) {
    .count = new_count,
    .buf = new_ptr
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} assign_operator_from_sb 'void (int, char *__single __sized_by(new_count))'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | | `-DependerDeclsAttr {{.+}} <<invalid sloc>> Implicit {{.+}} 0
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   |-DeclStmt {{.+}}
// CHECK-NEXT: |   | `-VarDecl {{.+}} used new 'struct sb'
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct sb' '='
// CHECK-NEXT: |     |-DeclRefExpr {{.+}} 'struct sb' lvalue Var {{.+}} 'new' 'struct sb'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct sb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct sb' lvalue
// CHECK-NEXT: |         `-BoundsCheckExpr {{.+}} 'struct sb' 'new_ptr <= __builtin_get_pointer_upper_bound(new_ptr) && __builtin_get_pointer_lower_bound(new_ptr) <= new_ptr && new_count <= (char *)__builtin_get_pointer_upper_bound(new_ptr) - (char *__bidi_indexable)new_ptr && 0 <= new_count'
// CHECK-NEXT: |           |-InitListExpr {{.+}} 'struct sb'
// CHECK-NEXT: |           | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |       | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |       | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |       | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |       | | `-<<<NULL>>>
// CHECK-NEXT: |           |       | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |       | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |       | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |       |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |       | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |       |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | | | | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | | |   `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           | | | |     |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           | | | |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | | |     | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | |     | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           | | | |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | | | |     | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | |     | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | | |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | | |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | | |     | | `-<<<NULL>>>
// CHECK-NEXT: |           | | | |     | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | |     | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | | |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | | |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | | |     |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | |     |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | | |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | | |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |           | | |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           | | |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           | | |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | |       | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |       | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           | | |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | | |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |       | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | |       | | `-<<<NULL>>>
// CHECK-NEXT: |           | | |       | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |       | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | |       |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |       |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           | |   |-GetBoundExpr {{.+}} 'char *' lower
// CHECK-NEXT: |           | |   | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |   |   `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           | |   |     |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           | |   |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |   |     | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |     | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           | |   |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | |   |     | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |     | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |   |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |   |     | | `-<<<NULL>>>
// CHECK-NEXT: |           | |   |     | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |     | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |   |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |   |     |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |     |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |   |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | |     `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |       `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           | |         |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           | |         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |         | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |         | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           | |         | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | |         | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |         | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |         | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |         | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |         | | `-<<<NULL>>>
// CHECK-NEXT: |           | |         | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |         | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |         |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |         |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |         |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |         |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |           `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |             `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: |           |   | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: |           |   |   |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK-NEXT: |           |   |   | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |           |   |   |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |   |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           |   |   |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           |   |   |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |   |       | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           |   |   |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |   |   |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   |   |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   |   |       | | `-<<<NULL>>>
// CHECK-NEXT: |           |   |   |       | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   |   |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   |   |       |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   |   |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   |   |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |   |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |           |   |       `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |         `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           |   |           |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           |   |           | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |           | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |           | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           |   |           | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |   |           | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |           | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   |           | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   |           | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   |           | | `-<<<NULL>>>
// CHECK-NEXT: |           |   |           | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |           | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   |           |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   |           |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   |           |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |           |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   |             `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   |               `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   `-BinaryOperator {{.+}} <<invalid sloc>, line:{{.+}}> 'int' '<='
// CHECK-NEXT: |           |     |-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |           |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |             `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |               |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |               | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |               | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |               | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |               | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |               | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |               | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |               | | `-<<<NULL>>>
// CHECK-NEXT: |               | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |               |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |               |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |               |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |                 `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |                   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
void assign_operator_from_sb(int new_count,
                             char* __sized_by(new_count) new_ptr) {
  struct sb new;
  new = (struct sb) {
    .count = new_count,
    .buf = new_ptr
  };
}


// CHECK-LABEL:|-FunctionDecl {{.+}} local_var_init_from_sb 'void (int, char *__single __sized_by(new_count))'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | | `-DependerDeclsAttr {{.+}} <<invalid sloc>> Implicit {{.+}} 0
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-DeclStmt {{.+}}
// CHECK-NEXT: |     `-VarDecl {{.+}} new 'struct sb' cinit
// CHECK-NEXT: |       `-ImplicitCastExpr {{.+}} 'struct sb' <LValueToRValue>
// CHECK-NEXT: |         `-CompoundLiteralExpr {{.+}} 'struct sb' lvalue
// CHECK-NEXT: |           `-BoundsCheckExpr {{.+}} 'struct sb' 'new_ptr <= __builtin_get_pointer_upper_bound(new_ptr) && __builtin_get_pointer_lower_bound(new_ptr) <= new_ptr && new_count <= (char *)__builtin_get_pointer_upper_bound(new_ptr) - (char *__bidi_indexable)new_ptr && 0 <= new_count'
// CHECK-NEXT: |             |-InitListExpr {{.+}} 'struct sb'
// CHECK-NEXT: |             | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |             | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |             | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |             | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |             |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |             |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |             |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |             |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |             |       | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |             |       | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |             |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |             |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |             |       | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |             |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |             |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |             |       | | `-<<<NULL>>>
// CHECK-NEXT: |             |       | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             |       | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |             |       | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |             |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |             |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |             |       |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             |       | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |             |       |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |             |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |             |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |             |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |             | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |             | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |             | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |             | | | | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |             | | | |   `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |             | | | |     |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |             | | | |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |             | | | |     | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             | | | |     | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |             | | | |     | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             | | | |     | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |             | | | |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |             | | | |     | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             | | | |     | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |             | | | |     | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             | | | |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |             | | | |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |             | | | |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |             | | | |     | | `-<<<NULL>>>
// CHECK-NEXT: |             | | | |     | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             | | | |     | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |             | | | |     | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             | | | |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |             | | | |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |             | | | |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |             | | | |     |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             | | | |     | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |             | | | |     |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             | | | |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |             | | | |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |             | | | |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |             | | | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |             | | |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |             | | |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |             | | |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |             | | |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |             | | |       | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             | | |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |             | | |       | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             | | |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |             | | |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |             | | |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             | | |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |             | | |       | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             | | |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |             | | |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |             | | |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |             | | |       | | `-<<<NULL>>>
// CHECK-NEXT: |             | | |       | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             | | |       | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |             | | |       | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             | | |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |             | | |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |             | | |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |             | | |       |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             | | |       | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |             | | |       |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             | | |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |             | | |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |             | | |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |             | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |             | |   |-GetBoundExpr {{.+}} 'char *' lower
// CHECK-NEXT: |             | |   | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |             | |   |   `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |             | |   |     |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |             | |   |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |             | |   |     | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             | |   |     | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |             | |   |     | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             | |   |     | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |             | |   |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |             | |   |     | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             | |   |     | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |             | |   |     | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             | |   |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |             | |   |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |             | |   |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |             | |   |     | | `-<<<NULL>>>
// CHECK-NEXT: |             | |   |     | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             | |   |     | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |             | |   |     | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             | |   |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |             | |   |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |             | |   |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |             | |   |     |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             | |   |     | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |             | |   |     |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             | |   |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |             | |   |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |             | |   |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |             | |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |             | |     `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |             | |       `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |             | |         |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |             | |         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |             | |         | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             | |         | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |             | |         | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             | |         | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |             | |         | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |             | |         | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             | |         | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |             | |         | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             | |         | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |             | |         | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |             | |         | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |             | |         | | `-<<<NULL>>>
// CHECK-NEXT: |             | |         | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             | |         | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |             | |         | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             | |         | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |             | |         |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |             | |         |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |             | |         |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             | |         | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |             | |         |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             | |         `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |             | |           `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |             | |             `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |             | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |             |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |             |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: |             |   | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |             |   | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |             |   | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |             |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: |             |   |   |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK-NEXT: |             |   |   | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |             |   |   |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |             |   |   |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |             |   |   |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |             |   |   |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |             |   |   |       | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             |   |   |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |             |   |   |       | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             |   |   |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |             |   |   |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |             |   |   |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             |   |   |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |             |   |   |       | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             |   |   |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |             |   |   |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |             |   |   |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |             |   |   |       | | `-<<<NULL>>>
// CHECK-NEXT: |             |   |   |       | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             |   |   |       | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |             |   |   |       | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             |   |   |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |             |   |   |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |             |   |   |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |             |   |   |       |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             |   |   |       | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |             |   |   |       |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             |   |   |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |             |   |   |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |             |   |   |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |             |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |             |   |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |             |   |       `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |             |   |         `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |             |   |           |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |             |   |           | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |             |   |           | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             |   |           | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |             |   |           | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             |   |           | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |             |   |           | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |             |   |           | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             |   |           | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |             |   |           | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             |   |           | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |             |   |           | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |             |   |           | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |             |   |           | | `-<<<NULL>>>
// CHECK-NEXT: |             |   |           | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             |   |           | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |             |   |           | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             |   |           | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |             |   |           |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |             |   |           |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |             |   |           |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             |   |           | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |             |   |           |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |             |   |           `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |             |   |             `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |             |   |               `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |             |   `-BinaryOperator {{.+}} <<invalid sloc>, line:{{.+}}> 'int' '<='
// CHECK-NEXT: |             |     |-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |             |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |             |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |             |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |             |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |             | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |             |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |             `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |               `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |                 |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |                 | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |                 | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |                 | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |                 | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |                 | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |                 | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |                 | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |                 | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |                 | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |                 | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |                 | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |                 | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |                 | | `-<<<NULL>>>
// CHECK-NEXT: |                 | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |                 | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |                 | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |                 | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |                 |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |                 |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |                 |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |                 | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |                 |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |                 `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |                   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |                     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
void local_var_init_from_sb(int new_count,
                            char* __sized_by(new_count) new_ptr) {
  struct sb new = (struct sb) {
    .count = new_count,
    .buf = new_ptr
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} call_arg_from_sb 'void (int, char *__single __sized_by(new_count))'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | | `-DependerDeclsAttr {{.+}} <<invalid sloc>> Implicit {{.+}} 0
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-CallExpr {{.+}} 'void'
// CHECK-NEXT: |     |-ImplicitCastExpr {{.+}} 'void (*__single)(struct sb)' <FunctionToPointerDecay>
// CHECK-NEXT: |     | `-DeclRefExpr {{.+}} 'void (struct sb)' Function {{.+}} 'consume_sb' 'void (struct sb)'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct sb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct sb' lvalue
// CHECK-NEXT: |         `-BoundsCheckExpr {{.+}} 'struct sb' 'new_ptr <= __builtin_get_pointer_upper_bound(new_ptr) && __builtin_get_pointer_lower_bound(new_ptr) <= new_ptr && new_count <= (char *)__builtin_get_pointer_upper_bound(new_ptr) - (char *__bidi_indexable)new_ptr && 0 <= new_count'
// CHECK-NEXT: |           |-InitListExpr {{.+}} 'struct sb'
// CHECK-NEXT: |           | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |       | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |       | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |       | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |       | | `-<<<NULL>>>
// CHECK-NEXT: |           |       | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |       | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |       | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |       |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |       | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |       |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | | | | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | | |   `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           | | | |     |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           | | | |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | | |     | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | |     | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           | | | |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | | | |     | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | |     | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | | |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | | |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | | |     | | `-<<<NULL>>>
// CHECK-NEXT: |           | | | |     | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | |     | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | | |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | | |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | | |     |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | |     |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | | |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | | |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |           | | |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           | | |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           | | |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | |       | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |       | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           | | |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | | |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |       | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | |       | | `-<<<NULL>>>
// CHECK-NEXT: |           | | |       | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |       | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | |       |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |       |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           | |   |-GetBoundExpr {{.+}} 'char *' lower
// CHECK-NEXT: |           | |   | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |   |   `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           | |   |     |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           | |   |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |   |     | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |     | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           | |   |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | |   |     | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |     | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |   |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |   |     | | `-<<<NULL>>>
// CHECK-NEXT: |           | |   |     | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |     | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |   |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |   |     |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |     |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |   |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | |     `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |       `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           | |         |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           | |         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |         | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |         | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           | |         | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | |         | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |         | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |         | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |         | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |         | | `-<<<NULL>>>
// CHECK-NEXT: |           | |         | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |         | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |         |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |         |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |         |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |         |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |           `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |             `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: |           |   | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: |           |   |   |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK-NEXT: |           |   |   | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |           |   |   |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |   |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           |   |   |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           |   |   |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |   |       | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           |   |   |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |   |   |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   |   |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   |   |       | | `-<<<NULL>>>
// CHECK-NEXT: |           |   |   |       | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   |   |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   |   |       |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   |   |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   |   |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |   |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |           |   |       `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |         `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           |   |           |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           |   |           | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |           | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |           | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           |   |           | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |   |           | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |           | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   |           | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   |           | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   |           | | `-<<<NULL>>>
// CHECK-NEXT: |           |   |           | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |           | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   |           |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   |           |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   |           |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |           |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   |             `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   |               `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   `-BinaryOperator {{.+}} <<invalid sloc>, line:{{.+}}> 'int' '<='
// CHECK-NEXT: |           |     |-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |           |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |             `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |               |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |               | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |               | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |               | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |               | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |               | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |               | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |               | | `-<<<NULL>>>
// CHECK-NEXT: |               | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |               |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |               |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |               |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |                 `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |                   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
void call_arg_from_sb(int new_count,
                      char* __sized_by(new_count) new_ptr) {
  consume_sb((struct sb) {
    .count = new_count,
    .buf = new_ptr
  });
}

// CHECK-LABEL:|-FunctionDecl {{.+}} return_sb_from_sb 'struct sb (int, char *__single __sized_by(new_count))'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | | `-DependerDeclsAttr {{.+}} <<invalid sloc>> Implicit {{.+}} 0
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-ReturnStmt {{.+}}
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct sb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct sb' lvalue
// CHECK-NEXT: |         `-BoundsCheckExpr {{.+}} 'struct sb' 'new_ptr <= __builtin_get_pointer_upper_bound(new_ptr) && __builtin_get_pointer_lower_bound(new_ptr) <= new_ptr && new_count <= (char *)__builtin_get_pointer_upper_bound(new_ptr) - (char *__bidi_indexable)new_ptr && 0 <= new_count'
// CHECK-NEXT: |           |-InitListExpr {{.+}} 'struct sb'
// CHECK-NEXT: |           | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |       | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |       | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |       | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |       | | `-<<<NULL>>>
// CHECK-NEXT: |           |       | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |       | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |       | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |       |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |       | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |       |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | | | | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | | |   `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           | | | |     |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           | | | |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | | |     | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | |     | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           | | | |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | | | |     | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | |     | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | | |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | | |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | | |     | | `-<<<NULL>>>
// CHECK-NEXT: |           | | | |     | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | |     | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | | |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | | |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | | |     |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | |     |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | | |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | | |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |           | | |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           | | |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           | | |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | |       | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |       | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           | | |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | | |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |       | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | |       | | `-<<<NULL>>>
// CHECK-NEXT: |           | | |       | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |       | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | |       |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |       |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           | |   |-GetBoundExpr {{.+}} 'char *' lower
// CHECK-NEXT: |           | |   | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |   |   `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           | |   |     |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           | |   |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |   |     | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |     | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           | |   |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | |   |     | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |     | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |   |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |   |     | | `-<<<NULL>>>
// CHECK-NEXT: |           | |   |     | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |     | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |   |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |   |     |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |     |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |   |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | |     `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |       `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           | |         |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           | |         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |         | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |         | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           | |         | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | |         | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |         | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |         | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |         | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |         | | `-<<<NULL>>>
// CHECK-NEXT: |           | |         | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |         | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |         |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |         |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |         |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |         |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |           `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |             `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: |           |   | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: |           |   |   |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK-NEXT: |           |   |   | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |           |   |   |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |   |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           |   |   |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           |   |   |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |   |       | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           |   |   |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |   |   |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   |   |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   |   |       | | `-<<<NULL>>>
// CHECK-NEXT: |           |   |   |       | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   |   |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   |   |       |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   |   |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   |   |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |   |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |           |   |       `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |         `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           |   |           |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           |   |           | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |           | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |           | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           |   |           | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |   |           | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |           | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   |           | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   |           | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   |           | | `-<<<NULL>>>
// CHECK-NEXT: |           |   |           | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |           | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   |           |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   |           |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   |           |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |           |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   |             `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   |               `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   `-BinaryOperator {{.+}} <<invalid sloc>, line:{{.+}}> 'int' '<='
// CHECK-NEXT: |           |     |-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |           |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |             `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |               |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |               | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |               | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |               | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |               | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |               | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |               | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |               | | `-<<<NULL>>>
// CHECK-NEXT: |               | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |               |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |               |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |               |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |                 `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |                   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
struct sb return_sb_from_sb(int new_count,
                            char* __sized_by(new_count) new_ptr) {
  return (struct sb) {
    .count = new_count,
    .buf = new_ptr
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} construct_not_used_from_sb 'void (int, char *__single __sized_by(new_count))'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | | `-DependerDeclsAttr {{.+}} <<invalid sloc>> Implicit {{.+}} 0
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-CStyleCastExpr {{.+}} 'void' <ToVoid>
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct sb' <LValueToRValue> part_of_explicit_cast
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct sb' lvalue
// CHECK-NEXT: |         `-BoundsCheckExpr {{.+}} 'struct sb' 'new_ptr <= __builtin_get_pointer_upper_bound(new_ptr) && __builtin_get_pointer_lower_bound(new_ptr) <= new_ptr && new_count <= (char *)__builtin_get_pointer_upper_bound(new_ptr) - (char *__bidi_indexable)new_ptr && 0 <= new_count'
// CHECK-NEXT: |           |-InitListExpr {{.+}} 'struct sb'
// CHECK-NEXT: |           | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |       | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |       | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |       | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |       | | `-<<<NULL>>>
// CHECK-NEXT: |           |       | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |       | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |       | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |       |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |       | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |       |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | | | | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | | |   `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           | | | |     |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           | | | |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | | |     | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | |     | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           | | | |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | | | |     | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | |     | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | | |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | | |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | | |     | | `-<<<NULL>>>
// CHECK-NEXT: |           | | | |     | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | |     | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | | |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | | |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | | |     |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | |     |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | | |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | | |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |           | | |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           | | |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           | | |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | |       | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |       | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           | | |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | | |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |       | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | |       | | `-<<<NULL>>>
// CHECK-NEXT: |           | | |       | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |       | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | |       |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |       |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           | |   |-GetBoundExpr {{.+}} 'char *' lower
// CHECK-NEXT: |           | |   | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |   |   `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           | |   |     |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           | |   |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |   |     | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |     | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           | |   |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | |   |     | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |     | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |   |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |   |     | | `-<<<NULL>>>
// CHECK-NEXT: |           | |   |     | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |     | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |   |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |   |     |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |     |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |   |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | |     `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |       `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           | |         |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           | |         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |         | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |         | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           | |         | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | |         | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |         | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |         | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |         | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |         | | `-<<<NULL>>>
// CHECK-NEXT: |           | |         | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |         | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |         |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |         |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |         |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |         |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |           `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |             `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: |           |   | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: |           |   |   |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK-NEXT: |           |   |   | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |           |   |   |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |   |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           |   |   |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           |   |   |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |   |       | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           |   |   |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |   |   |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   |   |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   |   |       | | `-<<<NULL>>>
// CHECK-NEXT: |           |   |   |       | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   |   |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   |   |       |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   |   |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   |   |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |   |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |           |   |       `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |         `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           |   |           |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           |   |           | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |           | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |           | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           |   |           | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |   |           | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |           | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   |           | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   |           | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   |           | | `-<<<NULL>>>
// CHECK-NEXT: |           |   |           | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |           | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   |           |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   |           |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   |           |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |           |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   |             `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   |               `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   `-BinaryOperator {{.+}} <<invalid sloc>, line:{{.+}}> 'int' '<='
// CHECK-NEXT: |           |     |-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |           |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |             `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |               |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |               | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |               | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |               | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |               | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |               | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |               | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |               | | `-<<<NULL>>>
// CHECK-NEXT: |               | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |               |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |               |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |               |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |                 `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |                   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
void construct_not_used_from_sb(int new_count,
                                char* __sized_by(new_count) new_ptr) {
  (void)(struct sb) {
    .count = new_count,
    .buf = new_ptr
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} assign_via_ptr_nested_from_sb 'void (struct nested_sb *__single, char *__single __sized_by(new_count), int)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'struct nested_sb *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | | `-DependerDeclsAttr {{.+}} <<invalid sloc>> Implicit {{.+}} 0
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct nested_sb' '='
// CHECK-NEXT: |     |-UnaryOperator {{.+}} 'struct nested_sb' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'struct nested_sb *__single' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.+}} 'struct nested_sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct nested_sb *__single'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct nested_sb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct nested_sb' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct nested_sb'
// CHECK-NEXT: |           |-BoundsCheckExpr {{.+}} 'struct sb' 'new_ptr <= __builtin_get_pointer_upper_bound(new_ptr) && __builtin_get_pointer_lower_bound(new_ptr) <= new_ptr && new_count <= (char *)__builtin_get_pointer_upper_bound(new_ptr) - (char *__bidi_indexable)new_ptr && 0 <= new_count'
// CHECK-NEXT: |           | |-InitListExpr {{.+}} 'struct sb'
// CHECK-NEXT: |           | | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           | |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           | |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |       | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |       | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           | |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |       | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |       | | `-<<<NULL>>>
// CHECK-NEXT: |           | |       | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |       | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |       | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |       |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |       | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |       |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           | | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           | | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           | | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | | | | | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | | | |   `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           | | | | |     |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           | | | | |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | | | |     | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | | |     | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | | |     | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | | |     | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           | | | | |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | | | | |     | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | | |     | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | | |     | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | | |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | | | |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | | | |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | | | |     | | `-<<<NULL>>>
// CHECK-NEXT: |           | | | | |     | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | | |     | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | | |     | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | | |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | | | |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | | | |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | | | |     |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | | |     | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | | |     |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | | |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | | | |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | | | |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | | | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |           | | | |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | | |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           | | | |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           | | | |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | | |       | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | |       | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           | | | |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | | | |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | |       | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | | |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | | |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | | |       | | `-<<<NULL>>>
// CHECK-NEXT: |           | | | |       | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |       | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | |       | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | | |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | | |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | | |       |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |       | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | |       |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | | |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | | |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           | | |   |-GetBoundExpr {{.+}} 'char *' lower
// CHECK-NEXT: |           | | |   | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | |   |   `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           | | |   |     |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           | | |   |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | |   |     | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |   |     | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |   |     | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |   |     | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           | | |   |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | | |   |     | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |   |     | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |   |     | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |   |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | |   |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | |   |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | |   |     | | `-<<<NULL>>>
// CHECK-NEXT: |           | | |   |     | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |   |     | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |   |     | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |   |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | |   |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | |   |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | |   |     |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |   |     | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |   |     |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |   |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | |   |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | |   |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | | |     `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | |       `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           | | |         |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           | | |         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | |         | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |         | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |         | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |         | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           | | |         | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | | |         | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |         | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |         | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |         | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | |         | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | |         | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | |         | | `-<<<NULL>>>
// CHECK-NEXT: |           | | |         | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |         | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |         | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |         | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | |         |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | |         |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | |         |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |         | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |         |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |         `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | |           `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | |             `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           | |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           | |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: |           | |   | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |   | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: |           | |   |   |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK-NEXT: |           | |   |   | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |           | |   |   |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |   |   |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           | |   |   |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           | |   |   |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |   |   |       | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |   |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |   |       | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |   |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           | |   |   |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | |   |   |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |   |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |   |       | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |   |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |   |   |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   |   |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |   |   |       | | `-<<<NULL>>>
// CHECK-NEXT: |           | |   |   |       | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |   |       | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |   |       | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |   |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |   |   |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   |   |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |   |   |       |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |   |       | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |   |       |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |   |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |   |   |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   |   |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | |   |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |           | |   |       `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |   |         `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           | |   |           |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           | |   |           | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |   |           | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |           | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |           | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |           | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           | |   |           | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | |   |           | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |           | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |           | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |           | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |   |           | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   |           | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |   |           | | `-<<<NULL>>>
// CHECK-NEXT: |           | |   |           | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |           | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |           | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |           | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |   |           |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   |           |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |   |           |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |           | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |           |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |           `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |   |             `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   |               `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |   `-BinaryOperator {{.+}} <<invalid sloc>, line:{{.+}}> 'int' '<='
// CHECK-NEXT: |           | |     |-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |           | |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           |     |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |     | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |     | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |     | | `-<<<NULL>>>
// CHECK-NEXT: |           |     | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |     |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           `-IntegerLiteral {{.+}} 'int' 0
void assign_via_ptr_nested_from_sb(struct nested_sb* ptr,
                                   char* __sized_by(new_count) new_ptr,
                                   int new_count) {
  *ptr = (struct nested_sb) {
    .nested = {.buf = new_ptr, .count = new_count },
    .other = 0x0
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} assign_via_ptr_nested_v2_from_sb 'void (struct nested_sb *__single, char *__single __sized_by(new_count), int)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'struct nested_sb *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | | `-DependerDeclsAttr {{.+}} <<invalid sloc>> Implicit {{.+}} 0
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct nested_sb' '='
// CHECK-NEXT: |     |-UnaryOperator {{.+}} 'struct nested_sb' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'struct nested_sb *__single' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.+}} 'struct nested_sb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct nested_sb *__single'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct nested_sb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct nested_sb' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct nested_sb'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'struct sb' <LValueToRValue>
// CHECK-NEXT: |           | `-CompoundLiteralExpr {{.+}} 'struct sb' lvalue
// CHECK-NEXT: |           |   `-BoundsCheckExpr {{.+}} 'struct sb' 'new_ptr <= __builtin_get_pointer_upper_bound(new_ptr) && __builtin_get_pointer_lower_bound(new_ptr) <= new_ptr && new_count <= (char *)__builtin_get_pointer_upper_bound(new_ptr) - (char *__bidi_indexable)new_ptr && 0 <= new_count'
// CHECK-NEXT: |           |     |-InitListExpr {{.+}} 'struct sb'
// CHECK-NEXT: |           |     | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |     | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |     | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |     | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |     |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |     |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           |     |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           |     |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |     |       | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     |       | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           |     |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |     |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     |       | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |     |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |     |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |     |       | | `-<<<NULL>>>
// CHECK-NEXT: |           |     |       | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     |       | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     |       | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |     |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |     |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |     |       |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     |       | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     |       |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |     |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |     |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |     |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           |     | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           |     | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |     | | | | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |     | | | |   `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           |     | | | |     |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           |     | | | |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |     | | | |     | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | | | |     | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     | | | |     | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | | | |     | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           |     | | | |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |     | | | |     | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | | | |     | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     | | | |     | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | | | |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |     | | | |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |     | | | |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |     | | | |     | | `-<<<NULL>>>
// CHECK-NEXT: |           |     | | | |     | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | | | |     | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     | | | |     | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | | | |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |     | | | |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |     | | | |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |     | | | |     |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | | | |     | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     | | | |     |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | | | |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |     | | | |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |     | | | |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |     | | | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |           |     | | |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |     | | |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           |     | | |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           |     | | |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |     | | |       | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | | |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     | | |       | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | | |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           |     | | |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |     | | |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | | |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     | | |       | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | | |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |     | | |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |     | | |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |     | | |       | | `-<<<NULL>>>
// CHECK-NEXT: |           |     | | |       | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | | |       | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     | | |       | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | | |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |     | | |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |     | | |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |     | | |       |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | | |       | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     | | |       |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | | |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |     | | |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |     | | |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |     | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           |     | |   |-GetBoundExpr {{.+}} 'char *' lower
// CHECK-NEXT: |           |     | |   | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |     | |   |   `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           |     | |   |     |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           |     | |   |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |     | |   |     | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | |   |     | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     | |   |     | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | |   |     | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           |     | |   |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |     | |   |     | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | |   |     | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     | |   |     | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | |   |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |     | |   |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |     | |   |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |     | |   |     | | `-<<<NULL>>>
// CHECK-NEXT: |           |     | |   |     | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | |   |     | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     | |   |     | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | |   |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |     | |   |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |     | |   |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |     | |   |     |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | |   |     | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     | |   |     |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | |   |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |     | |   |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |     | |   |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |     | |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |     | |     `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |     | |       `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           |     | |         |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           |     | |         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |     | |         | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | |         | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     | |         | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | |         | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           |     | |         | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |     | |         | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | |         | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     | |         | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | |         | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |     | |         | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |     | |         | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |     | |         | | `-<<<NULL>>>
// CHECK-NEXT: |           |     | |         | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | |         | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     | |         | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | |         | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |     | |         |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |     | |         |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |     | |         |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | |         | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     | |         |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | |         `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |     | |           `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |     | |             `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |     | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           |     |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           |     |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: |           |     |   | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |     |   | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |     |   | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |     |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: |           |     |   |   |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK-NEXT: |           |     |   |   | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |           |     |   |   |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |     |   |   |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           |     |   |   |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           |     |   |   |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |     |   |   |       | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     |   |   |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     |   |   |       | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     |   |   |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           |     |   |   |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |     |   |   |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     |   |   |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     |   |   |       | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     |   |   |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |     |   |   |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |     |   |   |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |     |   |   |       | | `-<<<NULL>>>
// CHECK-NEXT: |           |     |   |   |       | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     |   |   |       | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     |   |   |       | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     |   |   |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |     |   |   |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |     |   |   |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |     |   |   |       |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     |   |   |       | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     |   |   |       |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     |   |   |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |     |   |   |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |     |   |   |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |     |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |     |   |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |           |     |   |       `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |     |   |         `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           |     |   |           |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           |     |   |           | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |     |   |           | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     |   |           | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     |   |           | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     |   |           | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           |     |   |           | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |     |   |           | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     |   |           | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     |   |           | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     |   |           | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |     |   |           | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |     |   |           | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |     |   |           | | `-<<<NULL>>>
// CHECK-NEXT: |           |     |   |           | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     |   |           | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     |   |           | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     |   |           | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |     |   |           |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |     |   |           |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |     |   |           |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     |   |           | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     |   |           |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     |   |           `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |     |   |             `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |     |   |               `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |     |   `-BinaryOperator {{.+}} <<invalid sloc>, line:{{.+}}> 'int' '<='
// CHECK-NEXT: |           |     |     |-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |           |     |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |     |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |     |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |     |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |     | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |     |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |     `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |       `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           |         |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           |         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |         | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |         | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |         | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |         | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           |         | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |         | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |         | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |         | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |         | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |         | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |         | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |         | | `-<<<NULL>>>
// CHECK-NEXT: |           |         | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |         | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |         | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |         | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |         |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |         |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |         |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |         | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |         |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |         `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |           `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |             `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           `-IntegerLiteral {{.+}} 'int' 0
void assign_via_ptr_nested_v2_from_sb(struct nested_sb* ptr,
                                      char* __sized_by(new_count)new_ptr,
                                      int new_count) {
  *ptr = (struct nested_sb) {
    .nested = (struct sb){.buf = new_ptr, .count = new_count },
    .other = 0x0
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} array_of_struct_init_from_sb 'void (char *__single __sized_by(new_count), int)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | | `-DependerDeclsAttr {{.+}} <<invalid sloc>> Implicit {{.+}} 0
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   |-DeclStmt {{.+}}
// CHECK-NEXT: |   | `-VarDecl {{.+}} used arr 'struct sb[2]' cinit
// CHECK-NEXT: |   |   `-CompoundLiteralExpr {{.+}} 'struct sb[2]'
// CHECK-NEXT: |   |     `-InitListExpr {{.+}} 'struct sb[2]'
// CHECK-NEXT: |   |       |-BoundsCheckExpr {{.+}} 'struct sb' 'new_ptr <= __builtin_get_pointer_upper_bound(new_ptr) && __builtin_get_pointer_lower_bound(new_ptr) <= new_ptr && new_count <= (char *)__builtin_get_pointer_upper_bound(new_ptr) - (char *__bidi_indexable)new_ptr && 0 <= new_count'
// CHECK-NEXT: |   |       | |-InitListExpr {{.+}} 'struct sb'
// CHECK-NEXT: |   |       | | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |       | | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |       | | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |   |       | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |       | |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |       | |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |       | |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |       | |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |       | |       | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |       | |       | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |   |       | |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |       | |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |       | |       | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |       | |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |       | |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |   |       | |       | | `-<<<NULL>>>
// CHECK-NEXT: |   |       | |       | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | |       | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |       | |       | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |       | |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |       | |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |   |       | |       |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | |       | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |       | |       |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |       | |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |       | |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |   |       | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |       | | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |       | | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |       | | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |       | | | | | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |       | | | | |   `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |       | | | | |     |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |       | | | | |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |       | | | | |     | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | | | | |     | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |       | | | | |     | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | | | | |     | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |   |       | | | | |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |       | | | | |     | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | | | | |     | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |       | | | | |     | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | | | | |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |       | | | | |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |       | | | | |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |   |       | | | | |     | | `-<<<NULL>>>
// CHECK-NEXT: |   |       | | | | |     | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | | | | |     | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |       | | | | |     | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | | | | |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |       | | | | |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |       | | | | |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |   |       | | | | |     |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | | | | |     | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |       | | | | |     |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | | | | |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |       | | | | |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |       | | | | |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |   |       | | | | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |   |       | | | |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |       | | | |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |       | | | |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |       | | | |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |       | | | |       | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | | | |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |       | | | |       | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | | | |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |   |       | | | |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |       | | | |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | | | |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |       | | | |       | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | | | |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |       | | | |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |       | | | |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |   |       | | | |       | | `-<<<NULL>>>
// CHECK-NEXT: |   |       | | | |       | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | | | |       | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |       | | | |       | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | | | |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |       | | | |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |       | | | |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |   |       | | | |       |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | | | |       | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |       | | | |       |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | | | |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |       | | | |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |       | | | |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |   |       | | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |       | | |   |-GetBoundExpr {{.+}} 'char *' lower
// CHECK-NEXT: |   |       | | |   | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |       | | |   |   `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |       | | |   |     |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |       | | |   |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |       | | |   |     | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | | |   |     | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |       | | |   |     | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | | |   |     | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |   |       | | |   |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |       | | |   |     | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | | |   |     | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |       | | |   |     | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | | |   |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |       | | |   |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |       | | |   |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |   |       | | |   |     | | `-<<<NULL>>>
// CHECK-NEXT: |   |       | | |   |     | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | | |   |     | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |       | | |   |     | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | | |   |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |       | | |   |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |       | | |   |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |   |       | | |   |     |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | | |   |     | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |       | | |   |     |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | | |   |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |       | | |   |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |       | | |   |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |   |       | | |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |       | | |     `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |       | | |       `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |       | | |         |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |       | | |         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |       | | |         | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | | |         | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |       | | |         | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | | |         | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |   |       | | |         | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |       | | |         | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | | |         | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |       | | |         | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | | |         | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |       | | |         | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |       | | |         | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |   |       | | |         | | `-<<<NULL>>>
// CHECK-NEXT: |   |       | | |         | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | | |         | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |       | | |         | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | | |         | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |       | | |         |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |       | | |         |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |   |       | | |         |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | | |         | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |       | | |         |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | | |         `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |       | | |           `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |       | | |             `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |   |       | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |       | |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |       | |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: |   |       | |   | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |       | |   | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |       | |   | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |   |       | |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: |   |       | |   |   |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK-NEXT: |   |       | |   |   | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |   |       | |   |   |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |       | |   |   |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |       | |   |   |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |       | |   |   |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |       | |   |   |       | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | |   |   |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |       | |   |   |       | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | |   |   |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |   |       | |   |   |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |       | |   |   |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | |   |   |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |       | |   |   |       | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | |   |   |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |       | |   |   |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |       | |   |   |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |   |       | |   |   |       | | `-<<<NULL>>>
// CHECK-NEXT: |   |       | |   |   |       | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | |   |   |       | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |       | |   |   |       | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | |   |   |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |       | |   |   |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |       | |   |   |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |   |       | |   |   |       |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | |   |   |       | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |       | |   |   |       |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | |   |   |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |       | |   |   |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |       | |   |   |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |   |       | |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |       | |   |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |   |       | |   |       `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |       | |   |         `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |       | |   |           |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |       | |   |           | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |       | |   |           | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | |   |           | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |       | |   |           | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | |   |           | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |   |       | |   |           | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |       | |   |           | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | |   |           | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |       | |   |           | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | |   |           | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |       | |   |           | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |       | |   |           | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |   |       | |   |           | | `-<<<NULL>>>
// CHECK-NEXT: |   |       | |   |           | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | |   |           | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |       | |   |           | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | |   |           | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |       | |   |           |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |       | |   |           |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |   |       | |   |           |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | |   |           | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |       | |   |           |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       | |   |           `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |       | |   |             `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |       | |   |               `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |   |       | |   `-BinaryOperator {{.+}} <<invalid sloc>, line:{{.+}}> 'int' '<='
// CHECK-NEXT: |   |       | |     |-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |   |       | |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |       | |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |       | |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |   |       | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |       | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |       | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |   |       | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |       |   `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |       |     |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |       |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |       |     | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       |     | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |       |     | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       |     | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |   |       |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |       |     | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       |     | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |       |     | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |       |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |       |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |   |       |     | | `-<<<NULL>>>
// CHECK-NEXT: |   |       |     | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       |     | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |       |     | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |       |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |       |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |   |       |     |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       |     | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |       |     |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |       |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |       |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |   |       `-BoundsCheckExpr {{.+}} 'struct sb' '0 == 0'
// CHECK-NEXT: |   |         |-InitListExpr {{.+}} 'struct sb'
// CHECK-NEXT: |   |         | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | | `-IntegerLiteral {{.+}} 'int' 0
// CHECK-NEXT: |   |         | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count)':'char *__single'
// CHECK-NEXT: |   |         |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <NullToPointer>
// CHECK-NEXT: |   |         |     `-IntegerLiteral {{.+}} 'int' 0
// CHECK-NEXT: |   |         |-BinaryOperator {{.+}} <col:{{.+}}, <invalid sloc>> 'int' '=='
// CHECK-NEXT: |   |         | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | | `-IntegerLiteral {{.+}} 'int' 0
// CHECK-NEXT: |   |         | `-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |   |         |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | `-IntegerLiteral {{.+}} 'int' 0
// CHECK-NEXT: |   |         `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count)':'char *__single'
// CHECK-NEXT: |   |           `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <NullToPointer>
// CHECK-NEXT: |   |             `-IntegerLiteral {{.+}} 'int' 0
// CHECK-NEXT: |   `-CallExpr {{.+}} 'void'
// CHECK-NEXT: |     |-ImplicitCastExpr {{.+}} 'void (*__single)(struct sb (*__single)[])' <FunctionToPointerDecay>
// CHECK-NEXT: |     | `-DeclRefExpr {{.+}} 'void (struct sb (*__single)[])' Function {{.+}} 'consume_sb_arr' 'void (struct sb (*__single)[])'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct sb (*__single)[]' <BoundsSafetyPointerCast>
// CHECK-NEXT: |       `-ImplicitCastExpr {{.+}} 'struct sb (*__bidi_indexable)[]' <BitCast>
// CHECK-NEXT: |         `-UnaryOperator {{.+}} 'struct sb (*__bidi_indexable)[2]' prefix '&' cannot overflow
// CHECK-NEXT: |           `-DeclRefExpr {{.+}} 'struct sb[2]' lvalue Var {{.+}} 'arr' 'struct sb[2]'
void array_of_struct_init_from_sb(char* __sized_by(new_count) new_ptr,
                                  int new_count) {
  struct sb arr[] = (struct sb[]) {
    {.count = new_count, .buf = new_ptr},
    {.count = 0x0, .buf = 0x0}
  };
  consume_sb_arr(&arr);
}

// CHECK-LABEL:|-FunctionDecl {{.+}} assign_via_ptr_other_data_side_effect_from_sb 'void (struct sb_with_other_data *__single, int, char *__single __sized_by(new_count))'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'struct sb_with_other_data *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | | `-DependerDeclsAttr {{.+}} <<invalid sloc>> Implicit {{.+}} 0
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct sb_with_other_data' '='
// CHECK-NEXT: |     |-UnaryOperator {{.+}} 'struct sb_with_other_data' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'struct sb_with_other_data *__single' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.+}} 'struct sb_with_other_data *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb_with_other_data *__single'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct sb_with_other_data' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct sb_with_other_data' lvalue
// CHECK-NEXT: |         `-BoundsCheckExpr {{.+}} 'struct sb_with_other_data' 'new_ptr <= __builtin_get_pointer_upper_bound(new_ptr) && __builtin_get_pointer_lower_bound(new_ptr) <= new_ptr && new_count <= (char *)__builtin_get_pointer_upper_bound(new_ptr) - (char *__bidi_indexable)new_ptr && 0 <= new_count'
// CHECK-NEXT: |           |-InitListExpr {{.+}} 'struct sb_with_other_data'
// CHECK-NEXT: |           | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |   `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           | |     |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           | |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |     | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |     | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |     | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |     | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           | |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | |     | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |     | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |     | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |     | | `-<<<NULL>>>
// CHECK-NEXT: |           | |     | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |     | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |     | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |     |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |     | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |     |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | `-CallExpr {{.+}} 'int'
// CHECK-NEXT: |           |   `-ImplicitCastExpr {{.+}} 'int (*__single)(void)' <FunctionToPointerDecay>
// CHECK-NEXT: |           |     `-DeclRefExpr {{.+}} 'int (void)' Function {{.+}} 'get_int' 'int (void)'
// CHECK-NEXT: |           |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | | | | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | | |   `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           | | | |     |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           | | | |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | | |     | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | |     | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           | | | |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | | | |     | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | |     | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | | |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | | |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | | |     | | `-<<<NULL>>>
// CHECK-NEXT: |           | | | |     | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | |     | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | | |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | | |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | | |     |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | | |     |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | | |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | | |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | | |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |           | | |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           | | |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           | | |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | | |       | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |       | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           | | |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | | |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |       | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | |       | | `-<<<NULL>>>
// CHECK-NEXT: |           | | |       | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |       | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | |       |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | | |       |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | | |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           | |   |-GetBoundExpr {{.+}} 'char *' lower
// CHECK-NEXT: |           | |   | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |   |   `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           | |   |     |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           | |   |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |   |     | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |     | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           | |   |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | |   |     | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |     | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |   |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |   |     | | `-<<<NULL>>>
// CHECK-NEXT: |           | |   |     | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |     | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |   |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |   |     |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   |     |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |   |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |   |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | |     `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |       `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           | |         |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           | |         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |         | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |         | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           | |         | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | |         | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |         | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |         | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |         | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |         | | `-<<<NULL>>>
// CHECK-NEXT: |           | |         | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |         | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |         |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |         |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |         |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |         |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           | |         `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | |           `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |             `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |           |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |           |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: |           |   | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: |           |   |   |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK-NEXT: |           |   |   | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |           |   |   |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |   |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           |   |   |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           |   |   |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |   |       | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           |   |   |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |   |   |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   |   |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   |   |       | | `-<<<NULL>>>
// CHECK-NEXT: |           |   |   |       | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   |   |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   |   |       |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |   |       |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |   |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   |   |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   |   |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |   |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |           |   |       `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |         `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           |   |           |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           |   |           | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |           | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |           | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           |   |           | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |   |           | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |           | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   |           | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   |           | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   |           | | `-<<<NULL>>>
// CHECK-NEXT: |           |   |           | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |           | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   |           |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   |           |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   |           |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |           |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   |           `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   |             `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   |               `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   `-BinaryOperator {{.+}} <<invalid sloc>, line:{{.+}}> 'int' '<='
// CHECK-NEXT: |           |     |-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |           |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |             `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |               |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |               | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |               | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |               | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |               | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |               | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |               | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |               | | `-<<<NULL>>>
// CHECK-NEXT: |               | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |               |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |               |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |               |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: |               `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |                 `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |                   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
void assign_via_ptr_other_data_side_effect_from_sb(struct sb_with_other_data* ptr,
                                                   int new_count,
                                                   char* __sized_by(new_count) new_ptr) {
  *ptr = (struct sb_with_other_data) {
    .count = new_count,
    .buf = new_ptr,
    // Side effect. Generates a warning but it is suppressed for this test
    .other = get_int()
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} assign_via_ptr_other_data_side_effect_zero_ptr_from_sb 'void (struct sb_with_other_data *__single, int, char *__single __sized_by(new_count))'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'struct sb_with_other_data *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | | `-DependerDeclsAttr {{.+}} <<invalid sloc>> Implicit {{.+}} 0
// CHECK-NEXT: | |-ParmVarDecl {{.+}} new_ptr 'char *__single __sized_by(new_count)':'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct sb_with_other_data' '='
// CHECK-NEXT: |     |-UnaryOperator {{.+}} 'struct sb_with_other_data' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'struct sb_with_other_data *__single' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.+}} 'struct sb_with_other_data *__single' lvalue ParmVar {{.+}} 'ptr' 'struct sb_with_other_data *__single'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct sb_with_other_data' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct sb_with_other_data' lvalue
// CHECK-NEXT: |         `-BoundsCheckExpr {{.+}} 'struct sb_with_other_data' 'new_count == 0'
// CHECK-NEXT: |           |-InitListExpr {{.+}} 'struct sb_with_other_data'
// CHECK-NEXT: |           | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count)':'char *__single'
// CHECK-NEXT: |           | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <NullToPointer>
// CHECK-NEXT: |           | |   `-IntegerLiteral {{.+}} 'int' 0
// CHECK-NEXT: |           | `-CallExpr {{.+}} 'int'
// CHECK-NEXT: |           |   `-ImplicitCastExpr {{.+}} 'int (*__single)(void)' <FunctionToPointerDecay>
// CHECK-NEXT: |           |     `-DeclRefExpr {{.+}} 'int (void)' Function {{.+}} 'get_int' 'int (void)'
// CHECK-NEXT: |           |-BinaryOperator {{.+}} <line:{{.+}}, <invalid sloc>> 'int' '=='
// CHECK-NEXT: |           | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | `-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |           |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count)':'char *__single'
// CHECK-NEXT: |             `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <NullToPointer>
// CHECK-NEXT: |               `-IntegerLiteral {{.+}} 'int' 0
void assign_via_ptr_other_data_side_effect_zero_ptr_from_sb(struct sb_with_other_data* ptr,
                                           int new_count,
                                           char* __sized_by(new_count) new_ptr) {
  *ptr = (struct sb_with_other_data) {
    .count = new_count,
    .buf = 0x0,
    // Side effect. Generates a warning but it is suppressed for this test
    .other = get_int()
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} call_arg_transparent_union_from_sb 'void (int, char *__bidi_indexable)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__bidi_indexable'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-CallExpr {{.+}} 'void'
// CHECK-NEXT: |     |-ImplicitCastExpr {{.+}} 'void (*__single)(union TransparentUnion)' <FunctionToPointerDecay>
// CHECK-NEXT: |     | `-DeclRefExpr {{.+}} 'void (union TransparentUnion)' Function {{.+}} 'receive_transparent_union' 'void (union TransparentUnion)'
// CHECK-NEXT: |     `-CompoundLiteralExpr {{.+}} 'union TransparentUnion'
// CHECK-NEXT: |       `-InitListExpr {{.+}} 'union TransparentUnion' field Field {{.+}} 'sb' 'struct sb_with_other_data'
// CHECK-NEXT: |         `-ImplicitCastExpr {{.+}} 'struct sb_with_other_data' <LValueToRValue>
// CHECK-NEXT: |           `-CompoundLiteralExpr {{.+}} 'struct sb_with_other_data' lvalue
// CHECK-NEXT: |             `-BoundsCheckExpr {{.+}} 'struct sb_with_other_data' 'new_ptr <= __builtin_get_pointer_upper_bound(new_ptr) && __builtin_get_pointer_lower_bound(new_ptr) <= new_ptr && new_count <= (char *)__builtin_get_pointer_upper_bound(new_ptr) - (char *__bidi_indexable)new_ptr && 0 <= new_count'
// CHECK-NEXT: |               |-InitListExpr {{.+}} 'struct sb_with_other_data'
// CHECK-NEXT: |               | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |               | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |               | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |               | |-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |               | | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |               | |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |               | |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |               | `-IntegerLiteral {{.+}} 'int' 0
// CHECK-NEXT: |               |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |               | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |               | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |               | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |               | | | | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |               | | | |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |               | | | |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |               | | | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |               | | |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |               | | |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |               | | |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |               | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |               | |   |-GetBoundExpr {{.+}} 'char *' lower
// CHECK-NEXT: |               | |   | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |               | |   |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |               | |   |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |               | |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |               | |     `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |               | |       `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |               | |         `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |               | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |               |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |               |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: |               |   | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |               |   | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |               |   | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |               |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: |               |   |   |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK-NEXT: |               |   |   | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |               |   |   |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |               |   |   |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |               |   |   |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |               |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |               |   |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |               |   |       `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |               |   |         `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |               |   |           `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |               |   `-BinaryOperator {{.+}} <<invalid sloc>, line:{{.+}}> 'int' '<='
// CHECK-NEXT: |               |     |-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |               |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |               |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |               |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |               |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |               | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |               |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |               `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |                 `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |                   `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
void call_arg_transparent_union_from_sb(int new_count,
                                        char* __bidi_indexable new_ptr) {
  receive_transparent_union(
    (struct sb_with_other_data) {
      .count = new_count,
      .buf = new_ptr,
      .other = 0x0
    }
  );
}


// CHECK-LABEL:`-FunctionDecl {{.+}} call_arg_transparent_union_untransparently_from_sb 'void (int, char *__bidi_indexable)'
// CHECK-NEXT:   |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT:   |-ParmVarDecl {{.+}} used new_ptr 'char *__bidi_indexable'
// CHECK-NEXT:   `-CompoundStmt {{.+}}
// CHECK-NEXT:     `-CallExpr {{.+}} 'void'
// CHECK-NEXT:       |-ImplicitCastExpr {{.+}} 'void (*__single)(union TransparentUnion)' <FunctionToPointerDecay>
// CHECK-NEXT:       | `-DeclRefExpr {{.+}} 'void (union TransparentUnion)' Function {{.+}} 'receive_transparent_union' 'void (union TransparentUnion)'
// CHECK-NEXT:       `-ImplicitCastExpr {{.+}} 'union TransparentUnion' <LValueToRValue>
// CHECK-NEXT:         `-CompoundLiteralExpr {{.+}} 'union TransparentUnion' lvalue
// CHECK-NEXT:           `-InitListExpr {{.+}} 'union TransparentUnion' field Field {{.+}} 'sb' 'struct sb_with_other_data'
// CHECK-NEXT:             `-BoundsCheckExpr {{.+}} 'struct sb_with_other_data' 'new_ptr <= __builtin_get_pointer_upper_bound(new_ptr) && __builtin_get_pointer_lower_bound(new_ptr) <= new_ptr && new_count <= (char *)__builtin_get_pointer_upper_bound(new_ptr) - (char *__bidi_indexable)new_ptr && 0 <= new_count'
// CHECK-NEXT:               |-InitListExpr {{.+}} 'struct sb_with_other_data'
// CHECK-NEXT:               | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:               | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:               | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT:               | |-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT:               | | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT:               | |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT:               | |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT:               | `-IntegerLiteral {{.+}} 'int' 0
// CHECK-NEXT:               |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT:               | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT:               | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT:               | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT:               | | | | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT:               | | | |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT:               | | | |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT:               | | | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT:               | | |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT:               | | |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT:               | | |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT:               | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT:               | |   |-GetBoundExpr {{.+}} 'char *' lower
// CHECK-NEXT:               | |   | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT:               | |   |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT:               | |   |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT:               | |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT:               | |     `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT:               | |       `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT:               | |         `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT:               | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT:               |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT:               |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT:               |   | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:               |   | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:               |   | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT:               |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT:               |   |   |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK-NEXT:               |   |   | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT:               |   |   |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT:               |   |   |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT:               |   |   |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT:               |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT:               |   |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// CHECK-NEXT:               |   |       `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT:               |   |         `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT:               |   |           `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT:               |   `-BinaryOperator {{.+}} <<invalid sloc>, line:{{.+}}> 'int' '<='
// CHECK-NEXT:               |     |-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT:               |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:               |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:               |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT:               |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:               | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:               |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT:               `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT:                 `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT:                   `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
void call_arg_transparent_union_untransparently_from_sb(int new_count,
                                char* __bidi_indexable new_ptr) {
  receive_transparent_union(
    (union TransparentUnion) {
      .sb = {
        .count = new_count,
        .buf = new_ptr,
        .other = 0x0
      }
    }
  );
}
