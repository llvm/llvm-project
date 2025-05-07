// FileCheck lines automatically generated using make-ast-dump-check-v2.py

// RUN: %clang_cc1 -ast-dump -fbounds-safety -fno-bounds-safety-bringup-missing-checks=compound_literal_init -Wno-bounds-attributes-init-list-side-effect -Wno-bounds-safety-init-list-partial-null -verify %s | FileCheck %s
// RUN: %clang_cc1 -x objective-c -fexperimental-bounds-safety-objc -ast-dump -fbounds-safety -fno-bounds-safety-bringup-missing-checks=compound_literal_init -Wno-bounds-attributes-init-list-side-effect -Wno-bounds-safety-init-list-partial-null -verify %s | FileCheck %s

#include <ptrcheck.h>

// expected-no-diagnostics
struct eb {
  char* __ended_by(end) start;
  char* end;
};
// CHECK-LABEL:|-FunctionDecl {{.+}} used consume_eb 'void (struct eb)'
// CHECK-NEXT: | `-ParmVarDecl {{.+}} 'struct eb'
void consume_eb(struct eb);
// CHECK-LABEL:|-FunctionDecl {{.+}} used consume_eb_arr 'void (struct eb (*__single)[])'
// CHECK-NEXT: | `-ParmVarDecl {{.+}} arr 'struct eb (*__single)[]'
void consume_eb_arr(struct eb (*arr)[]);

struct nested_eb {
  struct eb nested;
  int other;
};

struct nested_and_outer_eb {
  struct eb nested;
  int other;
  char* __ended_by(end) start;
  char* end;
};


// CHECK-LABEL:|-FunctionDecl {{.+}} used get_int 'int (void)'
int get_int(void);

struct eb_with_other_data {
  char* __ended_by(end) start;
  char* end;
  int other;
};
struct no_attr_with_other_data {
  int count;
  char* buf;
  int other;
};
_Static_assert(sizeof(struct eb_with_other_data) == 
               sizeof(struct no_attr_with_other_data), "size mismatch");
// CHECK-LABEL:|-FunctionDecl {{.+}} consume_eb_with_other_data_arr 'void (struct eb_with_other_data (*__single)[])'
// CHECK-NEXT: | `-ParmVarDecl {{.+}} arr 'struct eb_with_other_data (*__single)[]'
void consume_eb_with_other_data_arr(struct eb_with_other_data (*arr)[]);

union TransparentUnion {
  struct eb_with_other_data eb;
  struct no_attr_with_other_data no_eb;
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

// CHECK-LABEL:|-FunctionDecl {{.+}} assign_via_ptr 'void (struct eb *__single, char *__bidi_indexable, char *__single)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'struct eb *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_start 'char *__bidi_indexable'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_end 'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct eb' '='
// CHECK-NEXT: |     |-UnaryOperator {{.+}} 'struct eb' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'struct eb *__single' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.+}} 'struct eb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct eb *__single'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct eb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct eb' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct eb'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           |   `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_start' 'char *__bidi_indexable'
// CHECK-NEXT: |           `-ImplicitCastExpr {{.+}} 'char *__single' <LValueToRValue>
// CHECK-NEXT: |             `-DeclRefExpr {{.+}} 'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single'
void assign_via_ptr(struct eb* ptr,
                    char* __bidi_indexable new_start, char* new_end) {
  *ptr = (struct eb) {
    .start = new_start,
    .end = new_end
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} assign_operator 'void (char *__bidi_indexable, char *__single)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_start 'char *__bidi_indexable'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_end 'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   |-DeclStmt {{.+}}
// CHECK-NEXT: |   | `-VarDecl {{.+}} used new 'struct eb'
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct eb' '='
// CHECK-NEXT: |     |-DeclRefExpr {{.+}} 'struct eb' lvalue Var {{.+}} 'new' 'struct eb'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct eb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct eb' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct eb'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           |   `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_start' 'char *__bidi_indexable'
// CHECK-NEXT: |           `-ImplicitCastExpr {{.+}} 'char *__single' <LValueToRValue>
// CHECK-NEXT: |             `-DeclRefExpr {{.+}} 'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single'
void assign_operator(char* __bidi_indexable new_start, char* new_end) {
  struct eb new;
  new = (struct eb) {
    .start = new_start,
    .end = new_end
  };
}


// CHECK-LABEL:|-FunctionDecl {{.+}} local_var_init 'void (char *__bidi_indexable, char *__single)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_start 'char *__bidi_indexable'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_end 'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-DeclStmt {{.+}}
// CHECK-NEXT: |     `-VarDecl {{.+}} new 'struct eb' cinit
// CHECK-NEXT: |       `-ImplicitCastExpr {{.+}} 'struct eb' <LValueToRValue>
// CHECK-NEXT: |         `-CompoundLiteralExpr {{.+}} 'struct eb' lvalue
// CHECK-NEXT: |           `-InitListExpr {{.+}} 'struct eb'
// CHECK-NEXT: |             |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |             | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |             |   `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_start' 'char *__bidi_indexable'
// CHECK-NEXT: |             `-ImplicitCastExpr {{.+}} 'char *__single' <LValueToRValue>
// CHECK-NEXT: |               `-DeclRefExpr {{.+}} 'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single'
void local_var_init(char* __bidi_indexable new_start, char* new_end) {
  struct eb new = (struct eb) {
    .start = new_start,
    .end = new_end
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} call_arg 'void (char *__bidi_indexable, char *__single)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_start 'char *__bidi_indexable'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_end 'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-CallExpr {{.+}} 'void'
// CHECK-NEXT: |     |-ImplicitCastExpr {{.+}} 'void (*__single)(struct eb)' <FunctionToPointerDecay>
// CHECK-NEXT: |     | `-DeclRefExpr {{.+}} 'void (struct eb)' Function {{.+}} 'consume_eb' 'void (struct eb)'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct eb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct eb' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct eb'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           |   `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_start' 'char *__bidi_indexable'
// CHECK-NEXT: |           `-ImplicitCastExpr {{.+}} 'char *__single' <LValueToRValue>
// CHECK-NEXT: |             `-DeclRefExpr {{.+}} 'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single'
void call_arg(char* __bidi_indexable new_start, char* new_end) {
  consume_eb((struct eb) {
    .start = new_start,
    .end = new_end
  });
}

// CHECK-LABEL:|-FunctionDecl {{.+}} return_eb 'struct eb (char *__bidi_indexable, char *__single)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_start 'char *__bidi_indexable'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_end 'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-ReturnStmt {{.+}}
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct eb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct eb' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct eb'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           |   `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_start' 'char *__bidi_indexable'
// CHECK-NEXT: |           `-ImplicitCastExpr {{.+}} 'char *__single' <LValueToRValue>
// CHECK-NEXT: |             `-DeclRefExpr {{.+}} 'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single'
struct eb return_eb(char* __bidi_indexable new_start, char* new_end) {
  return (struct eb) {
    .start = new_start,
    .end = new_end
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} construct_not_used 'void (char *__bidi_indexable, char *__single)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_start 'char *__bidi_indexable'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_end 'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-CStyleCastExpr {{.+}} 'void' <ToVoid>
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct eb' <LValueToRValue> part_of_explicit_cast
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct eb' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct eb'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           |   `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_start' 'char *__bidi_indexable'
// CHECK-NEXT: |           `-ImplicitCastExpr {{.+}} 'char *__single' <LValueToRValue>
// CHECK-NEXT: |             `-DeclRefExpr {{.+}} 'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single'
void construct_not_used(char* __bidi_indexable new_start, char* new_end) {
  (void)(struct eb) {
    .start = new_start,
    .end = new_end
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} assign_via_ptr_nullptr 'void (struct eb *__single, char *__single)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'struct eb *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_end 'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct eb' '='
// CHECK-NEXT: |     |-UnaryOperator {{.+}} 'struct eb' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'struct eb *__single' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.+}} 'struct eb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct eb *__single'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct eb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct eb' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct eb'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <NullToPointer>
// CHECK-NEXT: |           | `-IntegerLiteral {{.+}} 'int' 0
// CHECK-NEXT: |           `-ImplicitCastExpr {{.+}} 'char *__single' <LValueToRValue>
// CHECK-NEXT: |             `-DeclRefExpr {{.+}} 'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single'
void assign_via_ptr_nullptr(struct eb* ptr, char* new_end) {
  *ptr = (struct eb) {
    // Diagnostic emitted here but suppressed for this test case
    .start = 0x0,
    .end = new_end
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} assign_via_ptr_nested 'void (struct nested_eb *__single, char *__bidi_indexable, char *__single)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'struct nested_eb *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_start 'char *__bidi_indexable'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_end 'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct nested_eb' '='
// CHECK-NEXT: |     |-UnaryOperator {{.+}} 'struct nested_eb' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'struct nested_eb *__single' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.+}} 'struct nested_eb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct nested_eb *__single'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct nested_eb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct nested_eb' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct nested_eb'
// CHECK-NEXT: |           |-InitListExpr {{.+}} 'struct eb'
// CHECK-NEXT: |           | |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           | |   `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_start' 'char *__bidi_indexable'
// CHECK-NEXT: |           | `-ImplicitCastExpr {{.+}} 'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   `-DeclRefExpr {{.+}} 'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single'
// CHECK-NEXT: |           `-IntegerLiteral {{.+}} 'int' 0
void assign_via_ptr_nested(struct nested_eb* ptr,
                           char* __bidi_indexable new_start,
                           char* new_end) {
  *ptr = (struct nested_eb) {
    .nested = {.start = new_start, .end = new_end },
    .other = 0x0
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} assign_via_ptr_nested_v2 'void (struct nested_eb *__single, char *__bidi_indexable, char *__single)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'struct nested_eb *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_start 'char *__bidi_indexable'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_end 'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct nested_eb' '='
// CHECK-NEXT: |     |-UnaryOperator {{.+}} 'struct nested_eb' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'struct nested_eb *__single' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.+}} 'struct nested_eb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct nested_eb *__single'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct nested_eb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct nested_eb' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct nested_eb'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'struct eb' <LValueToRValue>
// CHECK-NEXT: |           | `-CompoundLiteralExpr {{.+}} 'struct eb' lvalue
// CHECK-NEXT: |           |   `-InitListExpr {{.+}} 'struct eb'
// CHECK-NEXT: |           |     |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |     | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           |     |   `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_start' 'char *__bidi_indexable'
// CHECK-NEXT: |           |     `-ImplicitCastExpr {{.+}} 'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |       `-DeclRefExpr {{.+}} 'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single'
// CHECK-NEXT: |           `-IntegerLiteral {{.+}} 'int' 0
void assign_via_ptr_nested_v2(struct nested_eb* ptr,
                              char* __bidi_indexable new_start,
                              char* new_end) {
  *ptr = (struct nested_eb) {
    .nested = (struct eb){.start = new_start, .end = new_end },
    .other = 0x0
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} assign_via_ptr_nested_v3 'void (struct nested_and_outer_eb *__single, char *__bidi_indexable, char *__single)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'struct nested_and_outer_eb *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_start 'char *__bidi_indexable'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_end 'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct nested_and_outer_eb' '='
// CHECK-NEXT: |     |-UnaryOperator {{.+}} 'struct nested_and_outer_eb' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'struct nested_and_outer_eb *__single' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.+}} 'struct nested_and_outer_eb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct nested_and_outer_eb *__single'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct nested_and_outer_eb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct nested_and_outer_eb' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct nested_and_outer_eb'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'struct eb' <LValueToRValue>
// CHECK-NEXT: |           | `-CompoundLiteralExpr {{.+}} 'struct eb' lvalue
// CHECK-NEXT: |           |   `-InitListExpr {{.+}} 'struct eb'
// CHECK-NEXT: |           |     |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |     | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           |     |   `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_start' 'char *__bidi_indexable'
// CHECK-NEXT: |           |     `-ImplicitCastExpr {{.+}} 'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |       `-DeclRefExpr {{.+}} 'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single'
// CHECK-NEXT: |           |-IntegerLiteral {{.+}} 'int' 0
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           |   `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_start' 'char *__bidi_indexable'
// CHECK-NEXT: |           `-ImplicitCastExpr {{.+}} 'char *__single' <LValueToRValue>
// CHECK-NEXT: |             `-DeclRefExpr {{.+}} 'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single'
void assign_via_ptr_nested_v3(struct nested_and_outer_eb* ptr,
                              char* __bidi_indexable new_start,
                              char* new_end) {
  *ptr = (struct nested_and_outer_eb) {
    .nested = (struct eb){.start = new_start, .end = new_end },
    .other = 0x0,
    .start = new_start,
    .end = new_end
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} array_of_struct_init 'void (char *__bidi_indexable, char *__single)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_start 'char *__bidi_indexable'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_end 'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   |-DeclStmt {{.+}}
// CHECK-NEXT: |   | `-VarDecl {{.+}} used arr 'struct eb[2]' cinit
// CHECK-NEXT: |   |   `-CompoundLiteralExpr {{.+}} 'struct eb[2]'
// CHECK-NEXT: |   |     `-InitListExpr {{.+}} 'struct eb[2]'
// CHECK-NEXT: |   |       |-InitListExpr {{.+}} 'struct eb'
// CHECK-NEXT: |   |       | |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |       | | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |       | |   `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_start' 'char *__bidi_indexable'
// CHECK-NEXT: |   |       | `-ImplicitCastExpr {{.+}} 'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |       |   `-DeclRefExpr {{.+}} 'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single'
// CHECK-NEXT: |   |       `-InitListExpr {{.+}} 'struct eb'
// CHECK-NEXT: |   |         |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <NullToPointer>
// CHECK-NEXT: |   |         | `-IntegerLiteral {{.+}} 'int' 0
// CHECK-NEXT: |   |         `-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(start) */ ':'char *__single' <NullToPointer>
// CHECK-NEXT: |   |           `-IntegerLiteral {{.+}} 'int' 0
// CHECK-NEXT: |   `-CallExpr {{.+}} 'void'
// CHECK-NEXT: |     |-ImplicitCastExpr {{.+}} 'void (*__single)(struct eb (*__single)[])' <FunctionToPointerDecay>
// CHECK-NEXT: |     | `-DeclRefExpr {{.+}} 'void (struct eb (*__single)[])' Function {{.+}} 'consume_eb_arr' 'void (struct eb (*__single)[])'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct eb (*__single)[]' <BoundsSafetyPointerCast>
// CHECK-NEXT: |       `-ImplicitCastExpr {{.+}} 'struct eb (*__bidi_indexable)[]' <BitCast>
// CHECK-NEXT: |         `-UnaryOperator {{.+}} 'struct eb (*__bidi_indexable)[2]' prefix '&' cannot overflow
// CHECK-NEXT: |           `-DeclRefExpr {{.+}} 'struct eb[2]' lvalue Var {{.+}} 'arr' 'struct eb[2]'
void array_of_struct_init(char* __bidi_indexable new_start, char* new_end) {
  struct eb arr[] = (struct eb[]) {
    {.start = new_start, .end = new_end},
    {.start = 0x0, .end = 0x0}
  };
  consume_eb_arr(&arr);
}


// CHECK-LABEL:|-FunctionDecl {{.+}} assign_via_ptr_other_data_side_effect 'void (struct eb_with_other_data *__single, char *__bidi_indexable, char *__single)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'struct eb_with_other_data *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_start 'char *__bidi_indexable'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_end 'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct eb_with_other_data' '='
// CHECK-NEXT: |     |-UnaryOperator {{.+}} 'struct eb_with_other_data' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'struct eb_with_other_data *__single' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.+}} 'struct eb_with_other_data *__single' lvalue ParmVar {{.+}} 'ptr' 'struct eb_with_other_data *__single'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct eb_with_other_data' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct eb_with_other_data' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct eb_with_other_data'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |           |   `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_start' 'char *__bidi_indexable'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | `-DeclRefExpr {{.+}} 'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single'
// CHECK-NEXT: |           `-CallExpr {{.+}} 'int'
// CHECK-NEXT: |             `-ImplicitCastExpr {{.+}} 'int (*__single)(void)' <FunctionToPointerDecay>
// CHECK-NEXT: |               `-DeclRefExpr {{.+}} 'int (void)' Function {{.+}} 'get_int' 'int (void)'
void assign_via_ptr_other_data_side_effect(struct eb_with_other_data* ptr,
                                           char* __bidi_indexable new_start,
                                           char* new_end) {
  *ptr = (struct eb_with_other_data) {
    .start = new_start,
    .end = new_end,
    // Side effect. Generates a warning but it is suppressed for this test
    .other = get_int()
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} assign_via_ptr_other_data_side_effect_zero_ptr 'void (struct eb_with_other_data *__single, char *__single)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'struct eb_with_other_data *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_end 'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct eb_with_other_data' '='
// CHECK-NEXT: |     |-UnaryOperator {{.+}} 'struct eb_with_other_data' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'struct eb_with_other_data *__single' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.+}} 'struct eb_with_other_data *__single' lvalue ParmVar {{.+}} 'ptr' 'struct eb_with_other_data *__single'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct eb_with_other_data' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct eb_with_other_data' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct eb_with_other_data'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <NullToPointer>
// CHECK-NEXT: |           | `-IntegerLiteral {{.+}} 'int' 0
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | `-DeclRefExpr {{.+}} 'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single'
// CHECK-NEXT: |           `-CallExpr {{.+}} 'int'
// CHECK-NEXT: |             `-ImplicitCastExpr {{.+}} 'int (*__single)(void)' <FunctionToPointerDecay>
// CHECK-NEXT: |               `-DeclRefExpr {{.+}} 'int (void)' Function {{.+}} 'get_int' 'int (void)'
void assign_via_ptr_other_data_side_effect_zero_ptr(struct eb_with_other_data* ptr,
                                                    char* new_end) {
  *ptr = (struct eb_with_other_data) {
    .start = 0x0,
    .end = new_end, // Generates a warning but it's suppressed in this test
    // Side effect. Generates a warning but it is suppressed for this test
    .other = get_int()
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} call_arg_transparent_union 'void (char *__bidi_indexable, char *__single)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_start 'char *__bidi_indexable'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_end 'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-CallExpr {{.+}} 'void'
// CHECK-NEXT: |     |-ImplicitCastExpr {{.+}} 'void (*__single)(union TransparentUnion)' <FunctionToPointerDecay>
// CHECK-NEXT: |     | `-DeclRefExpr {{.+}} 'void (union TransparentUnion)' Function {{.+}} 'receive_transparent_union' 'void (union TransparentUnion)'
// CHECK-NEXT: |     `-CompoundLiteralExpr {{.+}} 'union TransparentUnion'
// CHECK-NEXT: |       `-InitListExpr {{.+}} 'union TransparentUnion' field Field {{.+}} 'eb' 'struct eb_with_other_data'
// CHECK-NEXT: |         `-ImplicitCastExpr {{.+}} 'struct eb_with_other_data' <LValueToRValue>
// CHECK-NEXT: |           `-CompoundLiteralExpr {{.+}} 'struct eb_with_other_data' lvalue
// CHECK-NEXT: |             `-InitListExpr {{.+}} 'struct eb_with_other_data'
// CHECK-NEXT: |               |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |               | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |               |   `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_start' 'char *__bidi_indexable'
// CHECK-NEXT: |               |-ImplicitCastExpr {{.+}} 'char *__single' <LValueToRValue>
// CHECK-NEXT: |               | `-DeclRefExpr {{.+}} 'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single'
// CHECK-NEXT: |               `-IntegerLiteral {{.+}} 'int' 0
void call_arg_transparent_union(char* __bidi_indexable new_start,
                                char* new_end) {
  receive_transparent_union(
    (struct eb_with_other_data) {
      .start = new_start,
      .end = new_end,
      .other = 0x0
    }
  );
}

// CHECK-LABEL:|-FunctionDecl {{.+}} call_arg_transparent_union_untransparently 'void (char *__bidi_indexable, char *__single)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_start 'char *__bidi_indexable'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_end 'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-CallExpr {{.+}} 'void'
// CHECK-NEXT: |     |-ImplicitCastExpr {{.+}} 'void (*__single)(union TransparentUnion)' <FunctionToPointerDecay>
// CHECK-NEXT: |     | `-DeclRefExpr {{.+}} 'void (union TransparentUnion)' Function {{.+}} 'receive_transparent_union' 'void (union TransparentUnion)'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'union TransparentUnion' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'union TransparentUnion' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'union TransparentUnion' field Field {{.+}} 'eb' 'struct eb_with_other_data'
// CHECK-NEXT: |           `-InitListExpr {{.+}} 'struct eb_with_other_data'
// CHECK-NEXT: |             |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |             | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |             |   `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'new_start' 'char *__bidi_indexable'
// CHECK-NEXT: |             |-ImplicitCastExpr {{.+}} 'char *__single' <LValueToRValue>
// CHECK-NEXT: |             | `-DeclRefExpr {{.+}} 'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single'
// CHECK-NEXT: |             `-IntegerLiteral {{.+}} 'int' 0
void call_arg_transparent_union_untransparently(
  char* __bidi_indexable new_start,
  char* new_end) {
  receive_transparent_union(
    (union TransparentUnion) {
      .eb = {
        .start = new_start,
        .end = new_end,
        .other = 0x0
      }
    }
  );
}

// ============================================================================= 
// Tests with __ended_by source ptr
//
// These are less readable due to the BoundsSafetyPromotionExpr
// ============================================================================= 

// CHECK-LABEL:|-FunctionDecl {{.+}} assign_via_ptr_from_eb 'void (struct eb *__single, char *__single __ended_by(new_end), char *__single /* __started_by(new_start) */ )'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'struct eb *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_start 'char *__single __ended_by(new_end)':'char *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_end 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct eb' '='
// CHECK-NEXT: |     |-UnaryOperator {{.+}} 'struct eb' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'struct eb *__single' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.+}} 'struct eb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct eb *__single'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct eb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct eb' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct eb'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | `-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |-DeclRefExpr {{.+}} 'char *__single __ended_by(new_end)':'char *__single' lvalue ParmVar {{.+}} 'new_start' 'char *__single __ended_by(new_end)':'char *__single'
// CHECK-NEXT: |           |   |-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   | `-DeclRefExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT: |           |   `-ImplicitCastExpr {{.+}} 'char *__single __ended_by(new_end)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     `-DeclRefExpr {{.+}} 'char *__single __ended_by(new_end)':'char *__single' lvalue ParmVar {{.+}} 'new_start' 'char *__single __ended_by(new_end)':'char *__single'
// CHECK-NEXT: |           `-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(start) */ ':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |             `-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |               |-DeclRefExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT: |               |-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               | `-DeclRefExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT: |               `-ImplicitCastExpr {{.+}} <<invalid sloc>> 'char *__single __ended_by(new_end)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |                 `-DeclRefExpr {{.+}} <<invalid sloc>> 'char *__single __ended_by(new_end)':'char *__single' lvalue ParmVar {{.+}} 'new_start' 'char *__single __ended_by(new_end)':'char *__single'
void assign_via_ptr_from_eb(struct eb* ptr,
                    char* __ended_by(new_end) new_start, char* new_end) {
  *ptr = (struct eb) {
    .start = new_start,
    .end = new_end
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} assign_operator_from_eb 'void (char *__single __ended_by(new_end), char *__single /* __started_by(new_start) */ )'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_start 'char *__single __ended_by(new_end)':'char *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_end 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   |-DeclStmt {{.+}}
// CHECK-NEXT: |   | `-VarDecl {{.+}} used new 'struct eb'
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct eb' '='
// CHECK-NEXT: |     |-DeclRefExpr {{.+}} 'struct eb' lvalue Var {{.+}} 'new' 'struct eb'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct eb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct eb' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct eb'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | `-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |-DeclRefExpr {{.+}} 'char *__single __ended_by(new_end)':'char *__single' lvalue ParmVar {{.+}} 'new_start' 'char *__single __ended_by(new_end)':'char *__single'
// CHECK-NEXT: |           |   |-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   | `-DeclRefExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT: |           |   `-ImplicitCastExpr {{.+}} 'char *__single __ended_by(new_end)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     `-DeclRefExpr {{.+}} 'char *__single __ended_by(new_end)':'char *__single' lvalue ParmVar {{.+}} 'new_start' 'char *__single __ended_by(new_end)':'char *__single'
// CHECK-NEXT: |           `-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(start) */ ':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |             `-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |               |-DeclRefExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT: |               |-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               | `-DeclRefExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT: |               `-ImplicitCastExpr {{.+}} <<invalid sloc>> 'char *__single __ended_by(new_end)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |                 `-DeclRefExpr {{.+}} <<invalid sloc>> 'char *__single __ended_by(new_end)':'char *__single' lvalue ParmVar {{.+}} 'new_start' 'char *__single __ended_by(new_end)':'char *__single'
void assign_operator_from_eb(char* __ended_by(new_end) new_start, char* new_end) {
  struct eb new;
  new = (struct eb) {
    .start = new_start,
    .end = new_end
  };
}


// CHECK-LABEL:|-FunctionDecl {{.+}} local_var_init_from_eb 'void (char *__single __ended_by(new_end), char *__single /* __started_by(new_start) */ )'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_start 'char *__single __ended_by(new_end)':'char *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_end 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-DeclStmt {{.+}}
// CHECK-NEXT: |     `-VarDecl {{.+}} new 'struct eb' cinit
// CHECK-NEXT: |       `-ImplicitCastExpr {{.+}} 'struct eb' <LValueToRValue>
// CHECK-NEXT: |         `-CompoundLiteralExpr {{.+}} 'struct eb' lvalue
// CHECK-NEXT: |           `-InitListExpr {{.+}} 'struct eb'
// CHECK-NEXT: |             |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |             | `-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |             |   |-DeclRefExpr {{.+}} 'char *__single __ended_by(new_end)':'char *__single' lvalue ParmVar {{.+}} 'new_start' 'char *__single __ended_by(new_end)':'char *__single'
// CHECK-NEXT: |             |   |-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' <LValueToRValue>
// CHECK-NEXT: |             |   | `-DeclRefExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT: |             |   `-ImplicitCastExpr {{.+}} 'char *__single __ended_by(new_end)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |             |     `-DeclRefExpr {{.+}} 'char *__single __ended_by(new_end)':'char *__single' lvalue ParmVar {{.+}} 'new_start' 'char *__single __ended_by(new_end)':'char *__single'
// CHECK-NEXT: |             `-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(start) */ ':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |               `-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |                 |-DeclRefExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT: |                 |-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' <LValueToRValue>
// CHECK-NEXT: |                 | `-DeclRefExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT: |                 `-ImplicitCastExpr {{.+}} <<invalid sloc>> 'char *__single __ended_by(new_end)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |                   `-DeclRefExpr {{.+}} <<invalid sloc>> 'char *__single __ended_by(new_end)':'char *__single' lvalue ParmVar {{.+}} 'new_start' 'char *__single __ended_by(new_end)':'char *__single'
void local_var_init_from_eb(char* __ended_by(new_end) new_start, char* new_end) {
  struct eb new = (struct eb) {
    .start = new_start,
    .end = new_end
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} call_arg_from_eb 'void (char *__single __ended_by(new_end), char *__single /* __started_by(new_start) */ )'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_start 'char *__single __ended_by(new_end)':'char *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_end 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-CallExpr {{.+}} 'void'
// CHECK-NEXT: |     |-ImplicitCastExpr {{.+}} 'void (*__single)(struct eb)' <FunctionToPointerDecay>
// CHECK-NEXT: |     | `-DeclRefExpr {{.+}} 'void (struct eb)' Function {{.+}} 'consume_eb' 'void (struct eb)'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct eb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct eb' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct eb'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | `-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |-DeclRefExpr {{.+}} 'char *__single __ended_by(new_end)':'char *__single' lvalue ParmVar {{.+}} 'new_start' 'char *__single __ended_by(new_end)':'char *__single'
// CHECK-NEXT: |           |   |-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   | `-DeclRefExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT: |           |   `-ImplicitCastExpr {{.+}} 'char *__single __ended_by(new_end)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     `-DeclRefExpr {{.+}} 'char *__single __ended_by(new_end)':'char *__single' lvalue ParmVar {{.+}} 'new_start' 'char *__single __ended_by(new_end)':'char *__single'
// CHECK-NEXT: |           `-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(start) */ ':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |             `-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |               |-DeclRefExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT: |               |-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               | `-DeclRefExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT: |               `-ImplicitCastExpr {{.+}} <<invalid sloc>> 'char *__single __ended_by(new_end)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |                 `-DeclRefExpr {{.+}} <<invalid sloc>> 'char *__single __ended_by(new_end)':'char *__single' lvalue ParmVar {{.+}} 'new_start' 'char *__single __ended_by(new_end)':'char *__single'
void call_arg_from_eb(char* __ended_by(new_end) new_start, char* new_end) {
  consume_eb((struct eb) {
    .start = new_start,
    .end = new_end
  });
}

// CHECK-LABEL:|-FunctionDecl {{.+}} return_eb_from_eb 'struct eb (char *__single __ended_by(new_end), char *__single /* __started_by(new_start) */ )'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_start 'char *__single __ended_by(new_end)':'char *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_end 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-ReturnStmt {{.+}}
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct eb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct eb' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct eb'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | `-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |-DeclRefExpr {{.+}} 'char *__single __ended_by(new_end)':'char *__single' lvalue ParmVar {{.+}} 'new_start' 'char *__single __ended_by(new_end)':'char *__single'
// CHECK-NEXT: |           |   |-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   | `-DeclRefExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT: |           |   `-ImplicitCastExpr {{.+}} 'char *__single __ended_by(new_end)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     `-DeclRefExpr {{.+}} 'char *__single __ended_by(new_end)':'char *__single' lvalue ParmVar {{.+}} 'new_start' 'char *__single __ended_by(new_end)':'char *__single'
// CHECK-NEXT: |           `-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(start) */ ':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |             `-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |               |-DeclRefExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT: |               |-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               | `-DeclRefExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT: |               `-ImplicitCastExpr {{.+}} <<invalid sloc>> 'char *__single __ended_by(new_end)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |                 `-DeclRefExpr {{.+}} <<invalid sloc>> 'char *__single __ended_by(new_end)':'char *__single' lvalue ParmVar {{.+}} 'new_start' 'char *__single __ended_by(new_end)':'char *__single'
struct eb return_eb_from_eb(char* __ended_by(new_end) new_start, char* new_end) {
  return (struct eb) {
    .start = new_start,
    .end = new_end
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} construct_not_used_from_eb 'void (char *__single __ended_by(new_end), char *__single /* __started_by(new_start) */ )'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_start 'char *__single __ended_by(new_end)':'char *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_end 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-CStyleCastExpr {{.+}} 'void' <ToVoid>
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct eb' <LValueToRValue> part_of_explicit_cast
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct eb' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct eb'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | `-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |-DeclRefExpr {{.+}} 'char *__single __ended_by(new_end)':'char *__single' lvalue ParmVar {{.+}} 'new_start' 'char *__single __ended_by(new_end)':'char *__single'
// CHECK-NEXT: |           |   |-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   | `-DeclRefExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT: |           |   `-ImplicitCastExpr {{.+}} 'char *__single __ended_by(new_end)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     `-DeclRefExpr {{.+}} 'char *__single __ended_by(new_end)':'char *__single' lvalue ParmVar {{.+}} 'new_start' 'char *__single __ended_by(new_end)':'char *__single'
// CHECK-NEXT: |           `-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(start) */ ':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |             `-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |               |-DeclRefExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT: |               |-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               | `-DeclRefExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT: |               `-ImplicitCastExpr {{.+}} <<invalid sloc>> 'char *__single __ended_by(new_end)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |                 `-DeclRefExpr {{.+}} <<invalid sloc>> 'char *__single __ended_by(new_end)':'char *__single' lvalue ParmVar {{.+}} 'new_start' 'char *__single __ended_by(new_end)':'char *__single'
void construct_not_used_from_eb(char* __ended_by(new_end) new_start, char* new_end) {
  (void)(struct eb) {
    .start = new_start,
    .end = new_end
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} assign_via_ptr_nullptr_from_eb 'void (struct eb *__single, char *__single)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'struct eb *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_end 'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct eb' '='
// CHECK-NEXT: |     |-UnaryOperator {{.+}} 'struct eb' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'struct eb *__single' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.+}} 'struct eb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct eb *__single'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct eb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct eb' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct eb'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <NullToPointer>
// CHECK-NEXT: |           | `-IntegerLiteral {{.+}} 'int' 0
// CHECK-NEXT: |           `-ImplicitCastExpr {{.+}} 'char *__single' <LValueToRValue>
// CHECK-NEXT: |             `-DeclRefExpr {{.+}} 'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single'
void assign_via_ptr_nullptr_from_eb(struct eb* ptr, char* new_end) {
  *ptr = (struct eb) {
    // Diagnostic emitted here but suppressed for this test case
    .start = 0x0,
    .end = new_end
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} assign_via_ptr_nested_from_eb 'void (struct nested_eb *__single, char *__single __ended_by(new_end), char *__single /* __started_by(new_start) */ )'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'struct nested_eb *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_start 'char *__single __ended_by(new_end)':'char *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_end 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct nested_eb' '='
// CHECK-NEXT: |     |-UnaryOperator {{.+}} 'struct nested_eb' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'struct nested_eb *__single' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.+}} 'struct nested_eb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct nested_eb *__single'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct nested_eb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct nested_eb' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct nested_eb'
// CHECK-NEXT: |           |-InitListExpr {{.+}} 'struct eb'
// CHECK-NEXT: |           | |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | | `-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           | |   |-DeclRefExpr {{.+}} 'char *__single __ended_by(new_end)':'char *__single' lvalue ParmVar {{.+}} 'new_start' 'char *__single __ended_by(new_end)':'char *__single'
// CHECK-NEXT: |           | |   |-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |   | `-DeclRefExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT: |           | |   `-ImplicitCastExpr {{.+}} 'char *__single __ended_by(new_end)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | |     `-DeclRefExpr {{.+}} 'char *__single __ended_by(new_end)':'char *__single' lvalue ParmVar {{.+}} 'new_start' 'char *__single __ended_by(new_end)':'char *__single'
// CHECK-NEXT: |           | `-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(start) */ ':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |   `-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |     |-DeclRefExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT: |           |     |-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     | `-DeclRefExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT: |           |     `-ImplicitCastExpr {{.+}} <<invalid sloc>> 'char *__single __ended_by(new_end)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |       `-DeclRefExpr {{.+}} <<invalid sloc>> 'char *__single __ended_by(new_end)':'char *__single' lvalue ParmVar {{.+}} 'new_start' 'char *__single __ended_by(new_end)':'char *__single'
// CHECK-NEXT: |           `-IntegerLiteral {{.+}} 'int' 0
void assign_via_ptr_nested_from_eb(struct nested_eb* ptr,
                           char* __ended_by(new_end) new_start,
                           char* new_end) {
  *ptr = (struct nested_eb) {
    .nested = {.start = new_start, .end = new_end },
    .other = 0x0
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} assign_via_ptr_nested_v2_from_eb 'void (struct nested_eb *__single, char *__single __ended_by(new_end), char *__single /* __started_by(new_start) */ )'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'struct nested_eb *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_start 'char *__single __ended_by(new_end)':'char *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_end 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct nested_eb' '='
// CHECK-NEXT: |     |-UnaryOperator {{.+}} 'struct nested_eb' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'struct nested_eb *__single' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.+}} 'struct nested_eb *__single' lvalue ParmVar {{.+}} 'ptr' 'struct nested_eb *__single'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct nested_eb' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct nested_eb' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct nested_eb'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'struct eb' <LValueToRValue>
// CHECK-NEXT: |           | `-CompoundLiteralExpr {{.+}} 'struct eb' lvalue
// CHECK-NEXT: |           |   `-InitListExpr {{.+}} 'struct eb'
// CHECK-NEXT: |           |     |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |     | `-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |     |   |-DeclRefExpr {{.+}} 'char *__single __ended_by(new_end)':'char *__single' lvalue ParmVar {{.+}} 'new_start' 'char *__single __ended_by(new_end)':'char *__single'
// CHECK-NEXT: |           |     |   |-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     |   | `-DeclRefExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT: |           |     |   `-ImplicitCastExpr {{.+}} 'char *__single __ended_by(new_end)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     |     `-DeclRefExpr {{.+}} 'char *__single __ended_by(new_end)':'char *__single' lvalue ParmVar {{.+}} 'new_start' 'char *__single __ended_by(new_end)':'char *__single'
// CHECK-NEXT: |           |     `-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(start) */ ':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           |       `-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |         |-DeclRefExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT: |           |         |-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |         | `-DeclRefExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT: |           |         `-ImplicitCastExpr {{.+}} <<invalid sloc>> 'char *__single __ended_by(new_end)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |           `-DeclRefExpr {{.+}} <<invalid sloc>> 'char *__single __ended_by(new_end)':'char *__single' lvalue ParmVar {{.+}} 'new_start' 'char *__single __ended_by(new_end)':'char *__single'
// CHECK-NEXT: |           `-IntegerLiteral {{.+}} 'int' 0
void assign_via_ptr_nested_v2_from_eb(struct nested_eb* ptr,
                              char* __ended_by(new_end) new_start,
                              char* new_end) {
  *ptr = (struct nested_eb) {
    .nested = (struct eb){.start = new_start, .end = new_end },
    .other = 0x0
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} array_of_struct_init_from_eb 'void (char *__single __ended_by(new_end), char *__single /* __started_by(new_start) */ )'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_start 'char *__single __ended_by(new_end)':'char *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_end 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   |-DeclStmt {{.+}}
// CHECK-NEXT: |   | `-VarDecl {{.+}} used arr 'struct eb[2]' cinit
// CHECK-NEXT: |   |   `-CompoundLiteralExpr {{.+}} 'struct eb[2]'
// CHECK-NEXT: |   |     `-InitListExpr {{.+}} 'struct eb[2]'
// CHECK-NEXT: |   |       |-InitListExpr {{.+}} 'struct eb'
// CHECK-NEXT: |   |       | |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |       | | `-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |       | |   |-DeclRefExpr {{.+}} 'char *__single __ended_by(new_end)':'char *__single' lvalue ParmVar {{.+}} 'new_start' 'char *__single __ended_by(new_end)':'char *__single'
// CHECK-NEXT: |   |       | |   |-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |       | |   | `-DeclRefExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT: |   |       | |   `-ImplicitCastExpr {{.+}} 'char *__single __ended_by(new_end)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |       | |     `-DeclRefExpr {{.+}} 'char *__single __ended_by(new_end)':'char *__single' lvalue ParmVar {{.+}} 'new_start' 'char *__single __ended_by(new_end)':'char *__single'
// CHECK-NEXT: |   |       | `-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(start) */ ':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |       |   `-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |       |     |-DeclRefExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT: |   |       |     |-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |       |     | `-DeclRefExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT: |   |       |     `-ImplicitCastExpr {{.+}} <<invalid sloc>> 'char *__single __ended_by(new_end)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |       |       `-DeclRefExpr {{.+}} <<invalid sloc>> 'char *__single __ended_by(new_end)':'char *__single' lvalue ParmVar {{.+}} 'new_start' 'char *__single __ended_by(new_end)':'char *__single'
// CHECK-NEXT: |   |       `-InitListExpr {{.+}} 'struct eb'
// CHECK-NEXT: |   |         |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <NullToPointer>
// CHECK-NEXT: |   |         | `-IntegerLiteral {{.+}} 'int' 0
// CHECK-NEXT: |   |         `-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(start) */ ':'char *__single' <NullToPointer>
// CHECK-NEXT: |   |           `-IntegerLiteral {{.+}} 'int' 0
// CHECK-NEXT: |   `-CallExpr {{.+}} 'void'
// CHECK-NEXT: |     |-ImplicitCastExpr {{.+}} 'void (*__single)(struct eb (*__single)[])' <FunctionToPointerDecay>
// CHECK-NEXT: |     | `-DeclRefExpr {{.+}} 'void (struct eb (*__single)[])' Function {{.+}} 'consume_eb_arr' 'void (struct eb (*__single)[])'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct eb (*__single)[]' <BoundsSafetyPointerCast>
// CHECK-NEXT: |       `-ImplicitCastExpr {{.+}} 'struct eb (*__bidi_indexable)[]' <BitCast>
// CHECK-NEXT: |         `-UnaryOperator {{.+}} 'struct eb (*__bidi_indexable)[2]' prefix '&' cannot overflow
// CHECK-NEXT: |           `-DeclRefExpr {{.+}} 'struct eb[2]' lvalue Var {{.+}} 'arr' 'struct eb[2]'
void array_of_struct_init_from_eb(char* __ended_by(new_end) new_start, char* new_end) {
  struct eb arr[] = (struct eb[]) {
    {.start = new_start, .end = new_end},
    {.start = 0x0, .end = 0x0}
  };
  consume_eb_arr(&arr);
}


// CHECK-LABEL:|-FunctionDecl {{.+}} assign_via_ptr_other_data_side_effect_from_eb 'void (struct eb_with_other_data *__single, char *__single __ended_by(new_end), char *__single /* __started_by(new_start) */ )'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'struct eb_with_other_data *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_start 'char *__single __ended_by(new_end)':'char *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_end 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct eb_with_other_data' '='
// CHECK-NEXT: |     |-UnaryOperator {{.+}} 'struct eb_with_other_data' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'struct eb_with_other_data *__single' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.+}} 'struct eb_with_other_data *__single' lvalue ParmVar {{.+}} 'ptr' 'struct eb_with_other_data *__single'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct eb_with_other_data' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct eb_with_other_data' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct eb_with_other_data'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | `-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |-DeclRefExpr {{.+}} 'char *__single __ended_by(new_end)':'char *__single' lvalue ParmVar {{.+}} 'new_start' 'char *__single __ended_by(new_end)':'char *__single'
// CHECK-NEXT: |           |   |-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   | `-DeclRefExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT: |           |   `-ImplicitCastExpr {{.+}} 'char *__single __ended_by(new_end)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     `-DeclRefExpr {{.+}} 'char *__single __ended_by(new_end)':'char *__single' lvalue ParmVar {{.+}} 'new_start' 'char *__single __ended_by(new_end)':'char *__single'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(start) */ ':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |           | `-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |           |   |-DeclRefExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT: |           |   |-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |   | `-DeclRefExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT: |           |   `-ImplicitCastExpr {{.+}} <<invalid sloc>> 'char *__single __ended_by(new_end)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |           |     `-DeclRefExpr {{.+}} <<invalid sloc>> 'char *__single __ended_by(new_end)':'char *__single' lvalue ParmVar {{.+}} 'new_start' 'char *__single __ended_by(new_end)':'char *__single'
// CHECK-NEXT: |           `-CallExpr {{.+}} 'int'
// CHECK-NEXT: |             `-ImplicitCastExpr {{.+}} 'int (*__single)(void)' <FunctionToPointerDecay>
// CHECK-NEXT: |               `-DeclRefExpr {{.+}} 'int (void)' Function {{.+}} 'get_int' 'int (void)'
void assign_via_ptr_other_data_side_effect_from_eb(struct eb_with_other_data* ptr,
                                           char* __ended_by(new_end) new_start,
                                           char* new_end) {
  *ptr = (struct eb_with_other_data) {
    .start = new_start,
    .end = new_end,
    // Side effect. Generates a warning but it is suppressed for this test
    .other = get_int()
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} assign_via_ptr_other_data_side_effect_zero_ptr_from_eb 'void (struct eb_with_other_data *__single, char *__single)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'struct eb_with_other_data *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_end 'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-BinaryOperator {{.+}} 'struct eb_with_other_data' '='
// CHECK-NEXT: |     |-UnaryOperator {{.+}} 'struct eb_with_other_data' lvalue prefix '*' cannot overflow
// CHECK-NEXT: |     | `-ImplicitCastExpr {{.+}} 'struct eb_with_other_data *__single' <LValueToRValue>
// CHECK-NEXT: |     |   `-DeclRefExpr {{.+}} 'struct eb_with_other_data *__single' lvalue ParmVar {{.+}} 'ptr' 'struct eb_with_other_data *__single'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct eb_with_other_data' <LValueToRValue>
// CHECK-NEXT: |       `-CompoundLiteralExpr {{.+}} 'struct eb_with_other_data' lvalue
// CHECK-NEXT: |         `-InitListExpr {{.+}} 'struct eb_with_other_data'
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <NullToPointer>
// CHECK-NEXT: |           | `-IntegerLiteral {{.+}} 'int' 0
// CHECK-NEXT: |           |-ImplicitCastExpr {{.+}} 'char *__single' <LValueToRValue>
// CHECK-NEXT: |           | `-DeclRefExpr {{.+}} 'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single'
// CHECK-NEXT: |           `-CallExpr {{.+}} 'int'
// CHECK-NEXT: |             `-ImplicitCastExpr {{.+}} 'int (*__single)(void)' <FunctionToPointerDecay>
// CHECK-NEXT: |               `-DeclRefExpr {{.+}} 'int (void)' Function {{.+}} 'get_int' 'int (void)'
void assign_via_ptr_other_data_side_effect_zero_ptr_from_eb(struct eb_with_other_data* ptr,
                                                    char* new_end) {
  *ptr = (struct eb_with_other_data) {
    .start = 0x0,
    .end = new_end, // Generates a warning but it's suppressed in this test
    // Side effect. Generates a warning but it is suppressed for this test
    .other = get_int()
  };
}

// CHECK-LABEL:|-FunctionDecl {{.+}} call_arg_transparent_union_from_eb 'void (char *__single __ended_by(new_end), char *__single /* __started_by(new_start) */ )'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_start 'char *__single __ended_by(new_end)':'char *__single'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used new_end 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-CallExpr {{.+}} 'void'
// CHECK-NEXT: |     |-ImplicitCastExpr {{.+}} 'void (*__single)(union TransparentUnion)' <FunctionToPointerDecay>
// CHECK-NEXT: |     | `-DeclRefExpr {{.+}} 'void (union TransparentUnion)' Function {{.+}} 'receive_transparent_union' 'void (union TransparentUnion)'
// CHECK-NEXT: |     `-CompoundLiteralExpr {{.+}} 'union TransparentUnion'
// CHECK-NEXT: |       `-InitListExpr {{.+}} 'union TransparentUnion' field Field {{.+}} 'eb' 'struct eb_with_other_data'
// CHECK-NEXT: |         `-ImplicitCastExpr {{.+}} 'struct eb_with_other_data' <LValueToRValue>
// CHECK-NEXT: |           `-CompoundLiteralExpr {{.+}} 'struct eb_with_other_data' lvalue
// CHECK-NEXT: |             `-InitListExpr {{.+}} 'struct eb_with_other_data'
// CHECK-NEXT: |               |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |               | `-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |               |   |-DeclRefExpr {{.+}} 'char *__single __ended_by(new_end)':'char *__single' lvalue ParmVar {{.+}} 'new_start' 'char *__single __ended_by(new_end)':'char *__single'
// CHECK-NEXT: |               |   |-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               |   | `-DeclRefExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT: |               |   `-ImplicitCastExpr {{.+}} 'char *__single __ended_by(new_end)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               |     `-DeclRefExpr {{.+}} 'char *__single __ended_by(new_end)':'char *__single' lvalue ParmVar {{.+}} 'new_start' 'char *__single __ended_by(new_end)':'char *__single'
// CHECK-NEXT: |               |-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(start) */ ':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |               | `-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |               |   |-DeclRefExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT: |               |   |-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               |   | `-DeclRefExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT: |               |   `-ImplicitCastExpr {{.+}} <<invalid sloc>> 'char *__single __ended_by(new_end)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |               |     `-DeclRefExpr {{.+}} <<invalid sloc>> 'char *__single __ended_by(new_end)':'char *__single' lvalue ParmVar {{.+}} 'new_start' 'char *__single __ended_by(new_end)':'char *__single'
// CHECK-NEXT: |               `-IntegerLiteral {{.+}} 'int' 0
void call_arg_transparent_union_from_eb(char* __ended_by(new_end) new_start,
                                char* new_end) {
  receive_transparent_union(
    (struct eb_with_other_data) {
      .start = new_start,
      .end = new_end,
      .other = 0x0
    }
  );
}

// CHECK-LABEL:`-FunctionDecl {{.+}} call_arg_transparent_union_untransparently_from_eb 'void (char *__single __ended_by(new_end), char *__single /* __started_by(new_start) */ )'
// CHECK-NEXT:   |-ParmVarDecl {{.+}} used new_start 'char *__single __ended_by(new_end)':'char *__single'
// CHECK-NEXT:   |-ParmVarDecl {{.+}} used new_end 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT:   `-CompoundStmt {{.+}}
// CHECK-NEXT:     `-CallExpr {{.+}} 'void'
// CHECK-NEXT:       |-ImplicitCastExpr {{.+}} 'void (*__single)(union TransparentUnion)' <FunctionToPointerDecay>
// CHECK-NEXT:       | `-DeclRefExpr {{.+}} 'void (union TransparentUnion)' Function {{.+}} 'receive_transparent_union' 'void (union TransparentUnion)'
// CHECK-NEXT:       `-ImplicitCastExpr {{.+}} 'union TransparentUnion' <LValueToRValue>
// CHECK-NEXT:         `-CompoundLiteralExpr {{.+}} 'union TransparentUnion' lvalue
// CHECK-NEXT:           `-InitListExpr {{.+}} 'union TransparentUnion' field Field {{.+}} 'eb' 'struct eb_with_other_data'
// CHECK-NEXT:             `-InitListExpr {{.+}} 'struct eb_with_other_data'
// CHECK-NEXT:               |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT:               | `-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT:               |   |-DeclRefExpr {{.+}} 'char *__single __ended_by(new_end)':'char *__single' lvalue ParmVar {{.+}} 'new_start' 'char *__single __ended_by(new_end)':'char *__single'
// CHECK-NEXT:               |   |-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' <LValueToRValue>
// CHECK-NEXT:               |   | `-DeclRefExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT:               |   `-ImplicitCastExpr {{.+}} 'char *__single __ended_by(new_end)':'char *__single' <LValueToRValue>
// CHECK-NEXT:               |     `-DeclRefExpr {{.+}} 'char *__single __ended_by(new_end)':'char *__single' lvalue ParmVar {{.+}} 'new_start' 'char *__single __ended_by(new_end)':'char *__single'
// CHECK-NEXT:               |-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(start) */ ':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT:               | `-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT:               |   |-DeclRefExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT:               |   |-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' <LValueToRValue>
// CHECK-NEXT:               |   | `-DeclRefExpr {{.+}} 'char *__single /* __started_by(new_start) */ ':'char *__single' lvalue ParmVar {{.+}} 'new_end' 'char *__single /* __started_by(new_start) */ ':'char *__single'
// CHECK-NEXT:               |   `-ImplicitCastExpr {{.+}} <<invalid sloc>> 'char *__single __ended_by(new_end)':'char *__single' <LValueToRValue>
// CHECK-NEXT:               |     `-DeclRefExpr {{.+}} <<invalid sloc>> 'char *__single __ended_by(new_end)':'char *__single' lvalue ParmVar {{.+}} 'new_start' 'char *__single __ended_by(new_end)':'char *__single'
// CHECK-NEXT:               `-IntegerLiteral {{.+}} 'int' 0
void call_arg_transparent_union_untransparently_from_eb(
  char* __ended_by(new_end) new_start,
  char* new_end) {
  receive_transparent_union(
    (union TransparentUnion) {
      .eb = {
        .start = new_start,
        .end = new_end,
        .other = 0x0
      }
    }
  );
}
