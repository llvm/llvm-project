// FileCheck lines automatically generated using make-ast-dump-check-v2.py

// RUN: %clang_cc1 -ast-dump -fbounds-safety -fno-bounds-safety-bringup-missing-checks=compound_literal_init -Wno-bounds-attributes-init-list-side-effect -verify %s | FileCheck %s
// RUN: %clang_cc1 -x objective-c -fexperimental-bounds-safety-objc -ast-dump -fbounds-safety -fno-bounds-safety-bringup-missing-checks=compound_literal_init -Wno-bounds-attributes-init-list-side-effect -verify %s | FileCheck %s

#include <ptrcheck.h>

// expected-no-diagnostics
struct cb {
  int count;
  char* __counted_by(count) buf;
};
// CHECK-LABEL:|-FunctionDecl {{.+}} used consume_cb 'void (struct cb)'
// CHECK-NEXT: | `-ParmVarDecl {{.+}} 'struct cb'
void consume_cb(struct cb);
// CHECK-LABEL:|-FunctionDecl {{.+}} used consume_cb_arr 'void (struct cb (*__single)[])'
// CHECK-NEXT: | `-ParmVarDecl {{.+}} arr 'struct cb (*__single)[]'
void consume_cb_arr(struct cb (*arr)[]);

struct nested_cb {
  struct cb nested;
  int other;
};

struct nested_and_outer_cb {
  struct cb nested;
  int other;
  int count;
  char* __counted_by(count) buf;
};

// CHECK-LABEL:|-FunctionDecl {{.+}} used get_int 'int (void)'
int get_int(void);

struct cb_with_other_data {
  int count;
  char* __counted_by(count) buf;
  int other;
};
struct no_attr_with_other_data {
  int count;
  char* buf;
  int other;
};
_Static_assert(sizeof(struct cb_with_other_data) == 
               sizeof(struct no_attr_with_other_data), "size mismatch");
// CHECK-LABEL:|-FunctionDecl {{.+}} consume_cb_with_other_data_arr 'void (struct cb_with_other_data (*__single)[])'
// CHECK-NEXT: | `-ParmVarDecl {{.+}} arr 'struct cb_with_other_data (*__single)[]'
void consume_cb_with_other_data_arr(struct cb_with_other_data (*arr)[]);

union TransparentUnion {
  struct cb_with_other_data cb;
  struct no_attr_with_other_data no_cb;
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

// CHECK-LABEL:|-FunctionDecl {{.+}} assign_via_ptr 'void (struct cb *__single, int, char *__bidi_indexable)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'struct cb *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__bidi_indexable'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct cb' '='
// CHECK-NEXT: |     |-UnaryOperator {{.+}} 'struct cb' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'struct cb *__single' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.+}} 'struct cb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct cb *__single'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct cb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct cb' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct cb'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |             `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |               `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
void assign_via_ptr(struct cb* ptr, int new_count,
                    char* __bidi_indexable new_ptr) {
  *ptr = (struct cb) {
    .count = new_count,
    .buf = new_ptr
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} assign_operator 'void (int, char *__bidi_indexable)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__bidi_indexable'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   |-DeclStmt {{.+}}
// CHECK-NEXT: |   | `-VarDecl {{.+}} used new 'struct cb'
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct cb' '='
// CHECK-NEXT: |     |-DeclRefExpr {{.+}} 'struct cb' lvalue Var {{.+}} 'new' 'struct cb'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct cb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct cb' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct cb'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |             `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |               `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
void assign_operator(int new_count, char* __bidi_indexable new_ptr) {
  struct cb new;
  new = (struct cb) {
    .count = new_count,
    .buf = new_ptr
  };
}


// CHECK-LABEL:|-FunctionDecl {{.+}} local_var_init 'void (int, char *__bidi_indexable)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__bidi_indexable'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-DeclStmt {{.+}}
// CHECK-NEXT: |     `-VarDecl {{.+}} new 'struct cb' cinit
// CHECK-NEXT: |       `-ImplicitCastExpr {{.+}} 'struct cb' <LValueToRValue>
// CHECK-NEXT: |         `-CompoundLiteralExpr {{.+}} 'struct cb' lvalue
// CHECK-NEXT: |           `-InitListExpr {{.+}} 'struct cb'
// CHECK-NEXT: |             |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |             | `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |             `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |               `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |                 `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
void local_var_init(int new_count, char* __bidi_indexable new_ptr) {
  struct cb new = (struct cb) {
    .count = new_count,
    .buf = new_ptr
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} call_arg 'void (int, char *__bidi_indexable)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__bidi_indexable'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-CallExpr {{.+}} 'void'
// CHECK-NEXT: |     |-ImplicitCastExpr {{.+}} 'void (*__single)(struct cb)' <FunctionToPointerDecay>
// CHECK-NEXT: |     | `-DeclRefExpr {{.+}} 'void (struct cb)' Function {{.+}} 'consume_cb' 'void (struct cb)'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct cb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct cb' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct cb'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |             `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |               `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
void call_arg(int new_count, char* __bidi_indexable new_ptr) {
  consume_cb((struct cb) {
    .count = new_count,
    .buf = new_ptr
  });
}

// CHECK-LABEL:|-FunctionDecl {{.+}} return_cb 'struct cb (int, char *__bidi_indexable)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__bidi_indexable'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-ReturnStmt {{.+}}
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct cb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct cb' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct cb'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |             `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |               `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
struct cb return_cb(int new_count, char* __bidi_indexable new_ptr) {
  return (struct cb) {
    .count = new_count,
    .buf = new_ptr
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} construct_not_used 'void (int, char *__bidi_indexable)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__bidi_indexable'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-CStyleCastExpr {{.+}} 'void' <ToVoid>
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct cb' <LValueToRValue> part_of_explicit_cast
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct cb' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct cb'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |             `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |               `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
void construct_not_used(int new_count, char* __bidi_indexable new_ptr) {
  (void)(struct cb) {
    .count = new_count,
    .buf = new_ptr
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} assign_via_ptr_nullptr 'void (struct cb *__single, int)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'struct cb *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct cb' '='
// CHECK-NEXT: |     |-UnaryOperator {{.+}} 'struct cb' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'struct cb *__single' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.+}} 'struct cb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct cb *__single'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct cb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct cb' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct cb'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <NullToPointer>
// CHECK-NEXT: |             `-IntegerLiteral {{.+}} 'int' 0
void assign_via_ptr_nullptr(struct cb* ptr, int new_count) {
  *ptr = (struct cb) {
    .count = new_count,
    .buf = 0x0
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} assign_via_ptr_nested 'void (struct nested_cb *__single, char *__bidi_indexable, int)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'struct nested_cb *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__bidi_indexable'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct nested_cb' '='
// CHECK-NEXT: |     |-UnaryOperator {{.+}} 'struct nested_cb' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'struct nested_cb *__single' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.+}} 'struct nested_cb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct nested_cb *__single'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct nested_cb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct nested_cb' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct nested_cb'
// CHECK-NEXT: |           |-InitListExpr {{.+}} 'struct cb'
// CHECK-NEXT: |           | |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           `-IntegerLiteral {{.+}} 'int' 0
void assign_via_ptr_nested(struct nested_cb* ptr,
                           char* __bidi_indexable new_ptr,
                           int new_count) {
  *ptr = (struct nested_cb) {
    .nested = {.buf = new_ptr, .count = new_count },
    .other = 0x0
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} assign_via_ptr_nested_v2 'void (struct nested_cb *__single, char *__bidi_indexable, int)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'struct nested_cb *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__bidi_indexable'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct nested_cb' '='
// CHECK-NEXT: |     |-UnaryOperator {{.+}} 'struct nested_cb' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'struct nested_cb *__single' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.+}} 'struct nested_cb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct nested_cb *__single'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct nested_cb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct nested_cb' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct nested_cb'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'struct cb' <LValueToRValue>
// CHECK-NEXT: |           | `-CompoundLiteralExpr {{.+}} 'struct cb' lvalue
// CHECK-NEXT: |           |   `-InitListExpr {{.+}} 'struct cb'
// CHECK-NEXT: |           |     |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |     | `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |     `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |       `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           |         `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           `-IntegerLiteral {{.+}} 'int' 0
void assign_via_ptr_nested_v2(struct nested_cb* ptr,
                              char* __bidi_indexable new_ptr,
                              int new_count) {
  *ptr = (struct nested_cb) {
    .nested = (struct cb){.buf = new_ptr, .count = new_count },
    .other = 0x0
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} assign_via_ptr_nested_v3 'void (struct nested_and_outer_cb *__single, char *__bidi_indexable, int)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'struct nested_and_outer_cb *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__bidi_indexable'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct nested_and_outer_cb' '='
// CHECK-NEXT: |     |-UnaryOperator {{.+}} 'struct nested_and_outer_cb' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'struct nested_and_outer_cb *__single' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.+}} 'struct nested_and_outer_cb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct nested_and_outer_cb *__single'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct nested_and_outer_cb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct nested_and_outer_cb' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct nested_and_outer_cb'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'struct cb' <LValueToRValue>
// CHECK-NEXT: |           | `-CompoundLiteralExpr {{.+}} 'struct cb' lvalue
// CHECK-NEXT: |           |   `-InitListExpr {{.+}} 'struct cb'
// CHECK-NEXT: |           |     |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |     | `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |     `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |       `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           |         `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           |-IntegerLiteral {{.+}} 'int' 0
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |             `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |               `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
void assign_via_ptr_nested_v3(struct nested_and_outer_cb* ptr,
                              char* __bidi_indexable new_ptr,
                              int new_count) {
  *ptr = (struct nested_and_outer_cb) {
    .nested = (struct cb){.buf = new_ptr, .count = new_count },
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
// CHECK-NEXT: |   | `-VarDecl {{.+}} used arr 'struct cb[2]' cinit
// CHECK-NEXT: |   |   `-CompoundLiteralExpr {{.+}} 'struct cb[2]'
// CHECK-NEXT: |   |     `-InitListExpr {{.+}} 'struct cb[2]'
// CHECK-NEXT: |   |       |-InitListExpr {{.+}} 'struct cb'
// CHECK-NEXT: |   |       | |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |       | | `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |   |       | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |       |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |       |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |   |       `-InitListExpr {{.+}} 'struct cb'
// CHECK-NEXT: |   |         |-IntegerLiteral {{.+}} 'int' 0
// CHECK-NEXT: |   |         `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <NullToPointer>
// CHECK-NEXT: |   |           `-IntegerLiteral {{.+}} 'int' 0
// CHECK-NEXT: |   `-CallExpr {{.+}} 'void'
// CHECK-NEXT: |     |-ImplicitCastExpr {{.+}} 'void (*__single)(struct cb (*__single)[])' <FunctionToPointerDecay>
// CHECK-NEXT: |     | `-DeclRefExpr {{.+}} 'void (struct cb (*__single)[])' Function {{.+}} 'consume_cb_arr' 'void (struct cb (*__single)[])'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct cb (*__single)[]' <BoundsSafetyPointerCast>
// CHECK-NEXT: |       `-ImplicitCastExpr {{.+}} 'struct cb (*__bidi_indexable)[]' <BitCast>
// CHECK-NEXT: |         `-UnaryOperator {{.+}} 'struct cb (*__bidi_indexable)[2]' prefix '&' cannot overflow
// CHECK-NEXT: |           `-DeclRefExpr {{.+}} 'struct cb[2]' lvalue Var {{.+}} 'arr' 'struct cb[2]'
void array_of_struct_init(char* __bidi_indexable new_ptr,
                          int new_count) {
  struct cb arr[] = (struct cb[]) {
    {.count = new_count, .buf = new_ptr},
    // Note the type of `.buf`'s assignment here looks odd in the AST (child of
    // BoundsCheckExpr that should be materialized) because the compound literal
    // is used as the base. E.g.:
    // 'char *__single __counted_by((struct cb[2]){{.count = new_count, .buf = new_ptr}, {.count = 0, .buf = 0}}[1UL].count)':'char *__single'
    //
    // We could consider constant folding the expression
    // to `__counted_by(0)` but this doesn't seem to be necessary.
    {.count = 0x0, .buf = 0x0}
  };
  consume_cb_arr(&arr);
}


// CHECK-LABEL:|-FunctionDecl {{.+}} assign_via_ptr_other_data_side_effect 'void (struct cb_with_other_data *__single, int, char *__bidi_indexable)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'struct cb_with_other_data *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__bidi_indexable'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct cb_with_other_data' '='
// CHECK-NEXT: |     |-UnaryOperator {{.+}} 'struct cb_with_other_data' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'struct cb_with_other_data *__single' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.+}} 'struct cb_with_other_data *__single' lvalue ParmVar {{.+}} 'ptr' 'struct cb_with_other_data *__single'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct cb_with_other_data' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct cb_with_other_data' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct cb_with_other_data'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           |   `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |           `-CallExpr {{.+}} 'int'
// CHECK-NEXT: |             `-ImplicitCastExpr {{.+}} 'int (*__single)(void)' <FunctionToPointerDecay>
// CHECK-NEXT: |               `-DeclRefExpr {{.+}} 'int (void)' Function {{.+}} 'get_int' 'int (void)'
void assign_via_ptr_other_data_side_effect(struct cb_with_other_data* ptr,
                                           int new_count,
                                           char* __bidi_indexable new_ptr) {
  *ptr = (struct cb_with_other_data) {
    .count = new_count,
    .buf = new_ptr,
    // Side effect. Generates a warning but it is suppressed for this test
    .other = get_int()
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} assign_via_ptr_other_data_side_effect_zero_ptr 'void (struct cb_with_other_data *__single, int, char *__bidi_indexable)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'struct cb_with_other_data *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} new_ptr 'char *__bidi_indexable'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct cb_with_other_data' '='
// CHECK-NEXT: |     |-UnaryOperator {{.+}} 'struct cb_with_other_data' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'struct cb_with_other_data *__single' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.+}} 'struct cb_with_other_data *__single' lvalue ParmVar {{.+}} 'ptr' 'struct cb_with_other_data *__single'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct cb_with_other_data' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct cb_with_other_data' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct cb_with_other_data'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <NullToPointer>
// CHECK-NEXT: |           | `-IntegerLiteral {{.+}} 'int' 0
// CHECK-NEXT: |           `-CallExpr {{.+}} 'int'
// CHECK-NEXT: |             `-ImplicitCastExpr {{.+}} 'int (*__single)(void)' <FunctionToPointerDecay>
// CHECK-NEXT: |               `-DeclRefExpr {{.+}} 'int (void)' Function {{.+}} 'get_int' 'int (void)'
void assign_via_ptr_other_data_side_effect_zero_ptr(struct cb_with_other_data* ptr,
                                           int new_count,
                                           char* __bidi_indexable new_ptr) {
  *ptr = (struct cb_with_other_data) {
    .count = new_count,
    // FIXME: The type that `0x0` gets implicitly casted to is weird because its
    // using the CompoundLiteralExpr as a base. That base contains a side-effect
    // which means its not a valid counted_by expression. So far this hasn't
    // mattered.
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
// CHECK-NEXT: |       `-InitListExpr {{.+}} 'union TransparentUnion' field Field {{.+}} 'cb' 'struct cb_with_other_data'
// CHECK-NEXT: |         `-ImplicitCastExpr {{.+}} 'struct cb_with_other_data' <LValueToRValue>
// CHECK-NEXT: |           `-CompoundLiteralExpr {{.+}} 'struct cb_with_other_data' lvalue
// CHECK-NEXT: |             `-InitListExpr {{.+}} 'struct cb_with_other_data'
// CHECK-NEXT: |               |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |               | `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |               |-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |               | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |               |   `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |               `-IntegerLiteral {{.+}} 'int' 0
void call_arg_transparent_union(int new_count,
                                char* __bidi_indexable new_ptr) {
  receive_transparent_union(
    (struct cb_with_other_data) {
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
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'union TransparentUnion' field Field {{.+}} 'cb' 'struct cb_with_other_data'
// CHECK-NEXT: |           `-InitListExpr {{.+}} 'struct cb_with_other_data'
// CHECK-NEXT: |             |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |             | `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |             |-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |             | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |             |   `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |             `-IntegerLiteral {{.+}} 'int' 0
void call_arg_transparent_union_untransparently(int new_count,
                                char* __bidi_indexable new_ptr) {
  receive_transparent_union(
    (union TransparentUnion) {
      .cb = {
        .count = new_count,
        .buf = new_ptr,
        .other = 0x0
      }
    }
  );
}

// ============================================================================= 
// Tests with __counted_by source ptr
//
// These are less readable due to the BoundsSafetyPromotionExpr
// ============================================================================= 
// CHECK-LABEL:|-FunctionDecl {{.+}} assign_via_ptr_from_ptr 'void (struct cb *__single)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'struct cb *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct cb' '='
// CHECK-NEXT: |     |-UnaryOperator {{.+}} 'struct cb' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'struct cb *__single' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.+}} 'struct cb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct cb *__single'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct cb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct cb' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct cb'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | `-MemberExpr {{.+}} 'int' lvalue ->count {{.+}}
// CHECK-NEXT: |           |   `-ImplicitCastExpr {{.+}} 'struct cb *__single' <LValueToRValue>
// CHECK-NEXT: |           |     `-DeclRefExpr {{.+}} 'struct cb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct cb *__single'
// CHECK-NEXT: |           `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |             `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |               |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |               | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |               | | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT: |               | | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               | | |   `-MemberExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ->buf {{.+}}
// CHECK-NEXT: |               | | |     `-OpaqueValueExpr {{.+}} 'struct cb *__single'
// CHECK-NEXT: |               | | |       `-ImplicitCastExpr {{.+}} 'struct cb *__single' <LValueToRValue>
// CHECK-NEXT: |               | | |         `-DeclRefExpr {{.+}} 'struct cb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct cb *__single'
// CHECK-NEXT: |               | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |               | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |               | | | | `-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT: |               | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               | | | |     `-MemberExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ->buf {{.+}}
// CHECK-NEXT: |               | | | |       `-OpaqueValueExpr {{.+}} 'struct cb *__single'
// CHECK-NEXT: |               | | | |         `-ImplicitCastExpr {{.+}} 'struct cb *__single' <LValueToRValue>
// CHECK-NEXT: |               | | | |           `-DeclRefExpr {{.+}} 'struct cb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct cb *__single'
// CHECK-NEXT: |               | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |               | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |               | | |     `-MemberExpr {{.+}} 'int' lvalue ->count {{.+}}
// CHECK-NEXT: |               | | |       `-OpaqueValueExpr {{.+}} 'struct cb *__single'
// CHECK-NEXT: |               | | |         `-ImplicitCastExpr {{.+}} 'struct cb *__single' <LValueToRValue>
// CHECK-NEXT: |               | | |           `-DeclRefExpr {{.+}} 'struct cb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct cb *__single'
// CHECK-NEXT: |               | | `-<<<NULL>>>
// CHECK-NEXT: |               | |-OpaqueValueExpr {{.+}} 'struct cb *__single'
// CHECK-NEXT: |               | | `-ImplicitCastExpr {{.+}} 'struct cb *__single' <LValueToRValue>
// CHECK-NEXT: |               | |   `-DeclRefExpr {{.+}} 'struct cb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct cb *__single'
// CHECK-NEXT: |               | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |               | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |               | |   `-MemberExpr {{.+}} 'int' lvalue ->count {{.+}}
// CHECK-NEXT: |               | |     `-OpaqueValueExpr {{.+}} 'struct cb *__single'
// CHECK-NEXT: |               | |       `-ImplicitCastExpr {{.+}} 'struct cb *__single' <LValueToRValue>
// CHECK-NEXT: |               | |         `-DeclRefExpr {{.+}} 'struct cb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct cb *__single'
// CHECK-NEXT: |               | `-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT: |               |   `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               |     `-MemberExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ->buf {{.+}}
// CHECK-NEXT: |               |       `-OpaqueValueExpr {{.+}} 'struct cb *__single'
// CHECK-NEXT: |               |         `-ImplicitCastExpr {{.+}} 'struct cb *__single' <LValueToRValue>
// CHECK-NEXT: |               |           `-DeclRefExpr {{.+}} 'struct cb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct cb *__single'
// CHECK-NEXT: |               |-OpaqueValueExpr {{.+}} 'struct cb *__single'
// CHECK-NEXT: |               | `-ImplicitCastExpr {{.+}} 'struct cb *__single' <LValueToRValue>
// CHECK-NEXT: |               |   `-DeclRefExpr {{.+}} 'struct cb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct cb *__single'
// CHECK-NEXT: |               |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |               | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |               |   `-MemberExpr {{.+}} 'int' lvalue ->count {{.+}}
// CHECK-NEXT: |               |     `-OpaqueValueExpr {{.+}} 'struct cb *__single'
// CHECK-NEXT: |               |       `-ImplicitCastExpr {{.+}} 'struct cb *__single' <LValueToRValue>
// CHECK-NEXT: |               |         `-DeclRefExpr {{.+}} 'struct cb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct cb *__single'
// CHECK-NEXT: |               `-OpaqueValueExpr {{.+}} 'char *__single __counted_by(count)':'char *__single'
// CHECK-NEXT: |                 `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |                   `-MemberExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' lvalue ->buf {{.+}}
// CHECK-NEXT: |                     `-OpaqueValueExpr {{.+}} 'struct cb *__single'
// CHECK-NEXT: |                       `-ImplicitCastExpr {{.+}} 'struct cb *__single' <LValueToRValue>
// CHECK-NEXT: |                         `-DeclRefExpr {{.+}} 'struct cb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct cb *__single'
void assign_via_ptr_from_ptr(struct cb* ptr) {
  *ptr = (struct cb) {
    .count = ptr->count,
    .buf = ptr->buf
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} assign_via_ptr_from_cb 'void (struct cb *__single, int, char *__single __counted_by(new_count))'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'struct cb *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | | `-DependerDeclsAttr {{.+}} <<invalid sloc>> Implicit {{.+}} 0
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct cb' '='
// CHECK-NEXT: |     |-UnaryOperator {{.+}} 'struct cb' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'struct cb *__single' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.+}} 'struct cb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct cb *__single'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct cb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct cb' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct cb'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |             `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |               |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |               | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |               | | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               | | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |               | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |               | | | | `-OpaqueValueExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               | | | |     `-DeclRefExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |               | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |               | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |               | | `-<<<NULL>>>
// CHECK-NEXT: |               | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |               | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |               |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |               |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |               |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |               | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |               `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |                 `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |                   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
void assign_via_ptr_from_cb(struct cb* ptr, int new_count,
                            char* __counted_by(new_count) new_ptr) {
  *ptr = (struct cb) {
    .count = new_count,
    .buf = new_ptr
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} assign_operator_from_cb 'void (int, char *__single __counted_by(new_count))'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | | `-DependerDeclsAttr {{.+}} <<invalid sloc>> Implicit {{.+}} 0
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   |-DeclStmt {{.+}}
// CHECK-NEXT: |   | `-VarDecl {{.+}} used new 'struct cb'
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct cb' '='
// CHECK-NEXT: |     |-DeclRefExpr {{.+}} 'struct cb' lvalue Var {{.+}} 'new' 'struct cb'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct cb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct cb' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct cb'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |             `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |               |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |               | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |               | | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               | | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |               | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |               | | | | `-OpaqueValueExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               | | | |     `-DeclRefExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |               | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |               | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |               | | `-<<<NULL>>>
// CHECK-NEXT: |               | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |               | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |               |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |               |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |               |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |               | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |               `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |                 `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |                   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
void assign_operator_from_cb(int new_count,
                             char* __counted_by(new_count) new_ptr) {
  struct cb new;
  new = (struct cb) {
    .count = new_count,
    .buf = new_ptr
  };
}


// CHECK-LABEL:|-FunctionDecl {{.+}} local_var_init_from_cb 'void (int, char *__single __counted_by(new_count))'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | | `-DependerDeclsAttr {{.+}} <<invalid sloc>> Implicit {{.+}} 0
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-DeclStmt {{.+}}
// CHECK-NEXT: |     `-VarDecl {{.+}} new 'struct cb' cinit
// CHECK-NEXT: |       `-ImplicitCastExpr {{.+}} 'struct cb' <LValueToRValue>
// CHECK-NEXT: |         `-CompoundLiteralExpr {{.+}} 'struct cb' lvalue
// CHECK-NEXT: |           `-InitListExpr {{.+}} 'struct cb'
// CHECK-NEXT: |             |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |             | `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |             `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |               `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |                 |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |                 | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |                 | | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |                 | | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |                 | | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |                 | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |                 | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |                 | | | | `-OpaqueValueExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |                 | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |                 | | | |     `-DeclRefExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |                 | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |                 | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |                 | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |                 | | `-<<<NULL>>>
// CHECK-NEXT: |                 | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |                 | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |                 | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |                 | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |                 |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |                 |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |                 |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |                 | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |                 |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |                 `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |                   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |                     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
void local_var_init_from_cb(int new_count,
                            char* __counted_by(new_count) new_ptr) {
  struct cb new = (struct cb) {
    .count = new_count,
    .buf = new_ptr
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} call_arg_from_cb 'void (int, char *__single __counted_by(new_count))'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | | `-DependerDeclsAttr {{.+}} <<invalid sloc>> Implicit {{.+}} 0
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-CallExpr {{.+}} 'void'
// CHECK-NEXT: |     |-ImplicitCastExpr {{.+}} 'void (*__single)(struct cb)' <FunctionToPointerDecay>
// CHECK-NEXT: |     | `-DeclRefExpr {{.+}} 'void (struct cb)' Function {{.+}} 'consume_cb' 'void (struct cb)'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct cb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct cb' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct cb'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |             `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |               |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |               | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |               | | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               | | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |               | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |               | | | | `-OpaqueValueExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               | | | |     `-DeclRefExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |               | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |               | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |               | | `-<<<NULL>>>
// CHECK-NEXT: |               | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |               | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |               |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |               |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |               |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |               | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |               `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |                 `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |                   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
void call_arg_from_cb(int new_count,
                      char* __counted_by(new_count) new_ptr) {
  consume_cb((struct cb) {
    .count = new_count,
    .buf = new_ptr
  });
}

// CHECK-LABEL:|-FunctionDecl {{.+}} return_cb_from_cb 'struct cb (int, char *__single __counted_by(new_count))'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | | `-DependerDeclsAttr {{.+}} <<invalid sloc>> Implicit {{.+}} 0
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-ReturnStmt {{.+}}
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct cb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct cb' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct cb'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |             `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |               |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |               | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |               | | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               | | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |               | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |               | | | | `-OpaqueValueExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               | | | |     `-DeclRefExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |               | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |               | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |               | | `-<<<NULL>>>
// CHECK-NEXT: |               | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |               | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |               |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |               |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |               |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |               | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |               `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |                 `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |                   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
struct cb return_cb_from_cb(int new_count,
                            char* __counted_by(new_count) new_ptr) {
  return (struct cb) {
    .count = new_count,
    .buf = new_ptr
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} construct_not_used_from_cb 'void (int, char *__single __counted_by(new_count))'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | | `-DependerDeclsAttr {{.+}} <<invalid sloc>> Implicit {{.+}} 0
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-CStyleCastExpr {{.+}} 'void' <ToVoid>
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct cb' <LValueToRValue> part_of_explicit_cast
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct cb' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct cb'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |             `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |               |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |               | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |               | | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               | | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |               | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |               | | | | `-OpaqueValueExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               | | | |     `-DeclRefExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |               | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |               | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |               | | `-<<<NULL>>>
// CHECK-NEXT: |               | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |               | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |               | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |               |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |               |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |               |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |               | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |               `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |                 `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |                   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
void construct_not_used_from_cb(int new_count,
                                char* __counted_by(new_count) new_ptr) {
  (void)(struct cb) {
    .count = new_count,
    .buf = new_ptr
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} assign_via_ptr_nested_from_cb 'void (struct nested_cb *__single, char *__single __counted_by(new_count), int)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'struct nested_cb *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | | `-DependerDeclsAttr {{.+}} <<invalid sloc>> Implicit {{.+}} 0
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct nested_cb' '='
// CHECK-NEXT: |     |-UnaryOperator {{.+}} 'struct nested_cb' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'struct nested_cb *__single' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.+}} 'struct nested_cb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct nested_cb *__single'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct nested_cb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct nested_cb' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct nested_cb'
// CHECK-NEXT: |           |-InitListExpr {{.+}} 'struct cb'
// CHECK-NEXT: |           | |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | | `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |   `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           |     |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |     | | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     | | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |     | | | | `-OpaqueValueExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     | | | |     `-DeclRefExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |     | | `-<<<NULL>>>
// CHECK-NEXT: |           |     | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |     |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |           |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           `-IntegerLiteral {{.+}} 'int' 0
void assign_via_ptr_nested_from_cb(struct nested_cb* ptr,
                                   char* __counted_by(new_count) new_ptr,
                                   int new_count) {
  *ptr = (struct nested_cb) {
    .nested = {.buf = new_ptr, .count = new_count },
    .other = 0x0
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} assign_via_ptr_nested_v2_from_cb 'void (struct nested_cb *__single, char *__single __counted_by(new_count), int)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'struct nested_cb *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | | `-DependerDeclsAttr {{.+}} <<invalid sloc>> Implicit {{.+}} 0
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct nested_cb' '='
// CHECK-NEXT: |     |-UnaryOperator {{.+}} 'struct nested_cb' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'struct nested_cb *__single' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.+}} 'struct nested_cb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct nested_cb *__single'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct nested_cb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct nested_cb' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct nested_cb'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'struct cb' <LValueToRValue>
// CHECK-NEXT: |           | `-CompoundLiteralExpr {{.+}} 'struct cb' lvalue
// CHECK-NEXT: |           |   `-InitListExpr {{.+}} 'struct cb'
// CHECK-NEXT: |           |     |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |     | `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |     `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |       `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           |         |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           |         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |         | | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |           |         | | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |         | | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |           |         | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           |         | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |         | | | | `-OpaqueValueExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |           |         | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |         | | | |     `-DeclRefExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |           |         | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |         | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |         | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |         | | `-<<<NULL>>>
// CHECK-NEXT: |           |         | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |           |         | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |         | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |           |         | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |         |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |         |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |         |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |           |         | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |         |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |           |         `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |           `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |             `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           `-IntegerLiteral {{.+}} 'int' 0
void assign_via_ptr_nested_v2_from_cb(struct nested_cb* ptr,
                                      char* __counted_by(new_count)new_ptr,
                                      int new_count) {
  *ptr = (struct nested_cb) {
    .nested = (struct cb){.buf = new_ptr, .count = new_count },
    .other = 0x0
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} array_of_struct_init_from_cb 'void (char *__single __counted_by(new_count), int)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | | `-DependerDeclsAttr {{.+}} <<invalid sloc>> Implicit {{.+}} 0
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   |-DeclStmt {{.+}}
// CHECK-NEXT: |   | `-VarDecl {{.+}} used arr 'struct cb[2]' cinit
// CHECK-NEXT: |   |   `-CompoundLiteralExpr {{.+}} 'struct cb[2]'
// CHECK-NEXT: |   |     `-InitListExpr {{.+}} 'struct cb[2]'
// CHECK-NEXT: |   |       |-InitListExpr {{.+}} 'struct cb'
// CHECK-NEXT: |   |       | |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |       | | `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |   |       | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |       |   `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |       |     |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |       |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |       |     | | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       |     | | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |       |     | | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       |     | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |   |       |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |       |     | | | | `-OpaqueValueExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       |     | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |       |     | | | |     `-DeclRefExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |       |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |       |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |   |       |     | | `-<<<NULL>>>
// CHECK-NEXT: |   |       |     | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       |     | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |       |     | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |       |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |       |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |   |       |     |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       |     | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |       |     |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |   |       |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |       |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |       |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |   |       `-InitListExpr {{.+}} 'struct cb'
// CHECK-NEXT: |   |         |-IntegerLiteral {{.+}} 'int' 0
// CHECK-NEXT: |   |         `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <NullToPointer>
// CHECK-NEXT: |   |           `-IntegerLiteral {{.+}} 'int' 0
// CHECK-NEXT: |   `-CallExpr {{.+}} 'void'
// CHECK-NEXT: |     |-ImplicitCastExpr {{.+}} 'void (*__single)(struct cb (*__single)[])' <FunctionToPointerDecay>
// CHECK-NEXT: |     | `-DeclRefExpr {{.+}} 'void (struct cb (*__single)[])' Function {{.+}} 'consume_cb_arr' 'void (struct cb (*__single)[])'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct cb (*__single)[]' <BoundsSafetyPointerCast>
// CHECK-NEXT: |       `-ImplicitCastExpr {{.+}} 'struct cb (*__bidi_indexable)[]' <BitCast>
// CHECK-NEXT: |         `-UnaryOperator {{.+}} 'struct cb (*__bidi_indexable)[2]' prefix '&' cannot overflow
// CHECK-NEXT: |           `-DeclRefExpr {{.+}} 'struct cb[2]' lvalue Var {{.+}} 'arr' 'struct cb[2]'
void array_of_struct_init_from_cb(char* __counted_by(new_count) new_ptr,
                                  int new_count) {
  struct cb arr[] = (struct cb[]) {
    {.count = new_count, .buf = new_ptr},
    {.count = 0x0, .buf = 0x0}
  };
  consume_cb_arr(&arr);
}

// CHECK-LABEL:|-FunctionDecl {{.+}} assign_via_ptr_other_data_side_effect_from_cb 'void (struct cb_with_other_data *__single, int, char *__single __counted_by(new_count))'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'struct cb_with_other_data *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | | `-DependerDeclsAttr {{.+}} <<invalid sloc>> Implicit {{.+}} 0
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct cb_with_other_data' '='
// CHECK-NEXT: |     |-UnaryOperator {{.+}} 'struct cb_with_other_data' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'struct cb_with_other_data *__single' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.+}} 'struct cb_with_other_data *__single' lvalue ParmVar {{.+}} 'ptr' 'struct cb_with_other_data *__single'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct cb_with_other_data' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct cb_with_other_data' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct cb_with_other_data'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |           |   |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |           |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   | | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   | | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   | | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |           |   | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |   | | | | `-OpaqueValueExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   | | | |     `-DeclRefExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   | | `-<<<NULL>>>
// CHECK-NEXT: |           |   | |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   | |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |   |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |   |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |   |-OpaqueValueExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   |   `-DeclRefExpr {{.+}} 'char *__single __counted_by(new_count)':'char *__single' lvalue ParmVar {{.+}} 'new_ptr' 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: |           |   `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |           |     `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           |       `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           `-CallExpr {{.+}} 'int'
// CHECK-NEXT: |             `-ImplicitCastExpr {{.+}} 'int (*__single)(void)' <FunctionToPointerDecay>
// CHECK-NEXT: |               `-DeclRefExpr {{.+}} 'int (void)' Function {{.+}} 'get_int' 'int (void)'
void assign_via_ptr_other_data_side_effect_from_cb(struct cb_with_other_data* ptr,
                                                   int new_count,
                                                   char* __counted_by(new_count) new_ptr) {
  *ptr = (struct cb_with_other_data) {
    .count = new_count,
    .buf = new_ptr,
    // Side effect. Generates a warning but it is suppressed for this test
    .other = get_int()
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} assign_via_ptr_other_data_side_effect_zero_ptr_from_cb 'void (struct cb_with_other_data *__single, int, char *__single __counted_by(new_count))'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'struct cb_with_other_data *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | | `-DependerDeclsAttr {{.+}} <<invalid sloc>> Implicit {{.+}} 0
// CHECK-NEXT: | |-ParmVarDecl {{.+}} new_ptr 'char *__single __counted_by(new_count)':'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct cb_with_other_data' '='
// CHECK-NEXT: |     |-UnaryOperator {{.+}} 'struct cb_with_other_data' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'struct cb_with_other_data *__single' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.+}} 'struct cb_with_other_data *__single' lvalue ParmVar {{.+}} 'ptr' 'struct cb_with_other_data *__single'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct cb_with_other_data' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct cb_with_other_data' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct cb_with_other_data'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |           | `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <NullToPointer>
// CHECK-NEXT: |           | `-IntegerLiteral {{.+}} 'int' 0
// CHECK-NEXT: |           `-CallExpr {{.+}} 'int'
// CHECK-NEXT: |             `-ImplicitCastExpr {{.+}} 'int (*__single)(void)' <FunctionToPointerDecay>
// CHECK-NEXT: |               `-DeclRefExpr {{.+}} 'int (void)' Function {{.+}} 'get_int' 'int (void)'
void assign_via_ptr_other_data_side_effect_zero_ptr_from_cb(struct cb_with_other_data* ptr,
                                           int new_count,
                                           char* __counted_by(new_count) new_ptr) {
  *ptr = (struct cb_with_other_data) {
    .count = new_count,
    // FIXME: The type that `0x0` gets implicitly casted to is weird because its
    // using the CompoundLiteralExpr as a base. That base contains a side-effect
    // which means its not a valid counted_by expression. So far this hasn't
    // mattered.
    .buf = 0x0,
    // Side effect. Generates a warning but it is suppressed for this test
    .other = get_int()
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} call_arg_transparent_union_from_cb 'void (int, char *__bidi_indexable)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_ptr 'char *__bidi_indexable'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-CallExpr {{.+}} 'void'
// CHECK-NEXT: |     |-ImplicitCastExpr {{.+}} 'void (*__single)(union TransparentUnion)' <FunctionToPointerDecay>
// CHECK-NEXT: |     | `-DeclRefExpr {{.+}} 'void (union TransparentUnion)' Function {{.+}} 'receive_transparent_union' 'void (union TransparentUnion)'
// CHECK-NEXT: |     `-CompoundLiteralExpr {{.+}} 'union TransparentUnion'
// CHECK-NEXT: |       `-InitListExpr {{.+}} 'union TransparentUnion' field Field {{.+}} 'cb' 'struct cb_with_other_data'
// CHECK-NEXT: |         `-ImplicitCastExpr {{.+}} 'struct cb_with_other_data' <LValueToRValue>
// CHECK-NEXT: |           `-CompoundLiteralExpr {{.+}} 'struct cb_with_other_data' lvalue
// CHECK-NEXT: |             `-InitListExpr {{.+}} 'struct cb_with_other_data'
// CHECK-NEXT: |               |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |               | `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT: |               |-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |               | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |               |   `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |               `-IntegerLiteral {{.+}} 'int' 0
void call_arg_transparent_union_from_cb(int new_count,
                                        char* __bidi_indexable new_ptr) {
  receive_transparent_union(
    (struct cb_with_other_data) {
      .count = new_count,
      .buf = new_ptr,
      .other = 0x0
    }
  );
}


// CHECK-LABEL:`-FunctionDecl {{.+}} call_arg_transparent_union_untransparently_from_cb 'void (int, char *__bidi_indexable)'
// CHECK-NEXT:   |-ParmVarDecl {{.+}} used new_count 'int'
// CHECK-NEXT:   |-ParmVarDecl {{.+}} used new_ptr 'char *__bidi_indexable'
// CHECK-NEXT:   `-CompoundStmt {{.+}}
// CHECK-NEXT:     `-CallExpr {{.+}} 'void'
// CHECK-NEXT:       |-ImplicitCastExpr {{.+}} 'void (*__single)(union TransparentUnion)' <FunctionToPointerDecay>
// CHECK-NEXT:       | `-DeclRefExpr {{.+}} 'void (union TransparentUnion)' Function {{.+}} 'receive_transparent_union' 'void (union TransparentUnion)'
// CHECK-NEXT:       `-ImplicitCastExpr {{.+}} 'union TransparentUnion' <LValueToRValue>
// CHECK-NEXT:         `-CompoundLiteralExpr {{.+}} 'union TransparentUnion' lvalue
// CHECK-NEXT:           `-InitListExpr {{.+}} 'union TransparentUnion' field Field {{.+}} 'cb' 'struct cb_with_other_data'
// CHECK-NEXT:             `-InitListExpr {{.+}} 'struct cb_with_other_data'
// CHECK-NEXT:               |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:               | `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'new_count' 'int'
// CHECK-NEXT:               |-ImplicitCastExpr {{.+}} 'char *__single __counted_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT:               | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT:               |   `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_ptr' 'char *__bidi_indexable'
// CHECK-NEXT:               `-IntegerLiteral {{.+}} 'int' 0
void call_arg_transparent_union_untransparently_from_cb(int new_count,
                                char* __bidi_indexable new_ptr) {
  receive_transparent_union(
    (union TransparentUnion) {
      .cb = {
        .count = new_count,
        .buf = new_ptr,
        .other = 0x0
      }
    }
  );
}
