
// RUN: not %clang_cc1 -fsyntax-only -fbounds-safety -ast-dump %s | FileCheck %s
// RUN: not %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s | FileCheck %s

// Tests the correctness of applying bounds attributes to complex type specifiers
// as well as to what extent other attributes (represented by _Nullable) are retained.

#include "complex-typespecs-with-bounds.h"
#include <ptrcheck.h>

void typeoftypes() {
    typeof((long * _Nullable) 0) __single p1;
    typeof(typeof(bar) *) __single p2;
}

struct S {
    char * _Nullable f1;
};

void typeofexprs(struct S s) {
    typeof(foo()) __single p1;
    typeof(&foo) __single p2;
    typeof(&foo) __unsafe_indexable p3; // error: pointer cannot have more than one bound attribute

    typeof(bar) __single p4;
    typeof(&bar) __single p5; // error: pointer cannot have more than one bound attribute
    typeof(bar) * __single p6;
    typeof(bar[2]) * __single p7;
    typeof(&bar[2]) __single p8;
    typeof(&*bar) __single p9;

    typeof(s.f1) __bidi_indexable p10;
    typeof(*s.f1) * __bidi_indexable p11;
    typeof(&*s.f1) __unsafe_indexable p12;
}

typedef typeof(*bar) my_t;
typedef typeof(bar) my_ptr_t;
typedef typeof(*bar) * my_manual_ptr_t;

void typedefs_of_typeof() {
    my_t * __single p1;
    my_ptr_t __single p2;
    my_manual_ptr_t __single p3;
    my_manual_ptr_t __bidi_indexable p4;
    my_manual_ptr_t __unsafe_indexable p5;
}

void autotypes(char * _Nullable __single p) {
    __auto_type * __unsafe_indexable p1 = p;
    __auto_type * __unsafe_indexable p2 = &*p;
}

void typeofexpr_typeofexpr() {
    typeof(bar) p1;
    typeof(p1) __single p2;
}

void typeofexpr_typeoftype_typeofexpr() {
    typeof(typeof(bar)) p1;
    typeof(p1) __single p2;
}

void typeof_autotype1() {
    __auto_type p1 = bar;
    typeof(p1) __single p2;
}

void typeof_autotype2() {
    __auto_type * p1 = bar;
    typeof(p1) __single p2;
}

// CHECK: TranslationUnitDecl
// CHECK: |-FunctionDecl [[func_foo:0x[^ ]+]] {{.+}} foo
// CHECK: |-VarDecl [[var_bar:0x[^ ]+]]
// CHECK: |-FunctionDecl [[func_typeoftypes:0x[^ ]+]] {{.+}} typeoftypes
// CHECK: | `-CompoundStmt
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl [[var_p1:0x[^ ]+]]
// CHECK: |   `-DeclStmt
// CHECK: |     `-VarDecl [[var_p2:0x[^ ]+]]
// CHECK: |-RecordDecl
// CHECK: | `-FieldDecl
// CHECK: |-FunctionDecl [[func_typeofexprs:0x[^ ]+]] {{.+}} typeofexprs
// CHECK: | |-ParmVarDecl [[var_referenced:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl [[var_p1:0x[^ ]+]]
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl [[var_p2:0x[^ ]+]]
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl [[var_p3:0x[^ ]+]]
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl [[var_p4:0x[^ ]+]]
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl [[var_p5:0x[^ ]+]]
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl [[var_p6:0x[^ ]+]]
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl [[var_p7:0x[^ ]+]]
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl [[var_p8:0x[^ ]+]]
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl [[var_p9:0x[^ ]+]]
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl [[var_p10:0x[^ ]+]]
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl [[var_p11:0x[^ ]+]]
// CHECK: |   `-DeclStmt
// CHECK: |     `-VarDecl [[var_p12:0x[^ ]+]]
// CHECK: |-TypedefDecl
// CHECK: | `-TypeOfExprType
// CHECK: |   |-ParenExpr
// CHECK: |   | `-UnaryOperator {{.+}} cannot overflow
// CHECK: |   |   `-ImplicitCastExpr {{.+}} 'char * _Nullable':'char *' <LValueToRValue>
// CHECK: |   |     `-DeclRefExpr {{.+}} [[var_bar]]
// CHECK: |   `-BuiltinType
// CHECK: |-TypedefDecl
// CHECK: | `-TypeOfExprType
// CHECK: |   |-ParenExpr
// CHECK: |   | `-DeclRefExpr {{.+}} [[var_bar]]
// CHECK: |   `-AttributedType
// CHECK: |     `-AttributedType
// CHECK: |       `-PointerType
// CHECK: |         `-BuiltinType
// CHECK: |-TypedefDecl
// CHECK: | `-PointerType
// CHECK: |   `-TypeOfExprType
// CHECK: |     |-ParenExpr
// CHECK: |     | `-UnaryOperator {{.+}} cannot overflow
// CHECK: |     |   `-ImplicitCastExpr {{.+}} 'char * _Nullable':'char *' <LValueToRValue>
// CHECK: |     |     `-DeclRefExpr {{.+}} [[var_bar]]
// CHECK: |     `-BuiltinType
// CHECK: |-FunctionDecl [[func_typedefs_of_typeof:0x[^ ]+]] {{.+}} typedefs_of_typeof
// CHECK: | `-CompoundStmt
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl [[var_p1_1:0x[^ ]+]]
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl [[var_p2_1:0x[^ ]+]]
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl [[var_p3_1:0x[^ ]+]]
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl [[var_p4_1:0x[^ ]+]]
// CHECK: |   `-DeclStmt
// CHECK: |     `-VarDecl [[var_p5_1:0x[^ ]+]]
// CHECK: |-FunctionDecl [[func_autotypes:0x[^ ]+]] {{.+}} autotypes
// CHECK: | |-ParmVarDecl [[var_p:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl [[var_p1_2:0x[^ ]+]]
// CHECK: |   |   `-ImplicitCastExpr {{.+}} 'char *__unsafe_indexable' <BoundsSafetyPointerCast>
// CHECK: |   |     `-ImplicitCastExpr {{.+}} 'char *__single _Nullable':'char *__single' <LValueToRValue>
// CHECK: |   |       `-DeclRefExpr {{.+}} [[var_p]]
// CHECK: |   `-DeclStmt
// CHECK: |     `-VarDecl [[var_p2_2:0x[^ ]+]]
// CHECK: |       `-ImplicitCastExpr {{.+}} 'char *__unsafe_indexable' <BoundsSafetyPointerCast>
// CHECK: |         `-UnaryOperator {{.+}} cannot overflow
// CHECK: |           `-UnaryOperator {{.+}} cannot overflow
// CHECK: |             `-ImplicitCastExpr {{.+}} 'char *__single _Nullable':'char *__single' <LValueToRValue>
// CHECK: |               `-DeclRefExpr {{.+}} [[var_p]]
// CHECK: |-FunctionDecl [[func_typeofexpr_typeofexpr:0x[^ ]+]] {{.+}} typeofexpr_typeofexpr
// CHECK: | `-CompoundStmt
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl [[var_referenced_1:0x[^ ]+]]
// CHECK: |   `-DeclStmt
// CHECK: |     `-VarDecl [[var_p2_3:0x[^ ]+]]
// CHECK: |-FunctionDecl [[func_typeofexpr_typeoftype_typeofexpr:0x[^ ]+]] {{.+}} typeofexpr_typeoftype_typeofexpr
// CHECK: | `-CompoundStmt
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl [[var_referenced_2:0x[^ ]+]]
// CHECK: |   `-DeclStmt
// CHECK: |     `-VarDecl [[var_p2_4:0x[^ ]+]]
// CHECK: |-FunctionDecl [[func_typeof_autotype1:0x[^ ]+]] {{.+}} typeof_autotype1
// CHECK: | `-CompoundStmt
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl [[var_referenced_3:0x[^ ]+]]
// CHECK: |   |   `-ImplicitCastExpr {{.+}} 'char * _Nullable':'char *' <LValueToRValue>
// CHECK: |   |     `-DeclRefExpr {{.+}} [[var_bar]]
// CHECK: |   `-DeclStmt
// CHECK: |     `-VarDecl [[var_p2_5:0x[^ ]+]]
// CHECK: `-FunctionDecl [[func_typeof_autotype2:0x[^ ]+]] {{.+}} typeof_autotype2
// CHECK:   `-CompoundStmt
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl [[var_referenced_4:0x[^ ]+]]
// CHECK:     |   `-ImplicitCastExpr {{.+}} 'char * _Nullable':'char *' <LValueToRValue>
// CHECK:     |     `-DeclRefExpr {{.+}} [[var_bar]]
// CHECK:     `-DeclStmt
// CHECK:       `-VarDecl [[var_p2_6:0x[^ ]+]]

