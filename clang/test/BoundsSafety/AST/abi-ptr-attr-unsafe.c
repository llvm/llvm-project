

// RUN: %clang_cc1 -verify -include %S/Inputs/abi-ptr-attr-unsafe/mock-system-header.h -fbounds-safety %s
// RUN: %clang_cc1 -ast-dump -include %S/Inputs/abi-ptr-attr-unsafe/mock-system-header.h -fbounds-safety %s | FileCheck %s
// RUN: %clang_cc1 -verify -include %S/Inputs/abi-ptr-attr-unsafe/mock-system-header.h -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s
// RUN: %clang_cc1 -ast-dump -include %S/Inputs/abi-ptr-attr-unsafe/mock-system-header.h -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s | FileCheck %s
// expected-no-diagnostics

#include <ptrcheck.h>


// CHECK-LABEL: FUnspecified 'void (int *)' inline
// CHECK: |-ParmVarDecl {{.*}} used x 'int *'
// CHECK: `-CompoundStmt
// CHECK:   |-DeclStmt
// CHECK:   | `-VarDecl {{.*}} used y 'int *' cinit
// CHECK:   |   `-ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK:   |     `-DeclRefExpr {{.*}} 'int *' lvalue ParmVar {{.*}} 'x' 'int *'
// CHECK:   |-BinaryOperator {{.*}} 'int *' '='
// CHECK:   | |-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} 'y' 'int *'
// CHECK:   | `-BinaryOperator {{.*}} 'int *' '+'
// CHECK:   |   |-ImplicitCastExpr {{.*}} 'int *' <LValueToRValue>
// CHECK:   |   | `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} 'y' 'int *'
// CHECK:   |   `-IntegerLiteral {{.*}} 'int' 1
// CHECK:   `-DeclStmt
// CHECK:     `-VarDecl {{.*}} z 'int **' cinit
// CHECK:       `-ImplicitCastExpr {{.*}} 'int **' <BoundsSafetyPointerCast>
// CHECK:       `-UnaryOperator {{.*}} 'int **__bidi_indexable' prefix '&' cannot overflow
// CHECK:         `-DeclRefExpr {{.*}} 'int *' lvalue Var {{.*}} 'y' 'int *'

// CHECK-LABEL: FUnspecifiedInline 'void (int *__single)'
// CHECK: |-ParmVarDecl {{.*}} x 'int *__single'
// CHECK: `-CompoundStmt
// CHECK:   `-CallExpr
// CHECK:     |-ImplicitCastExpr {{.*}} 'void (*__single)(int *)' <FunctionToPointerDecay>
// CHECK:     | `-DeclRefExpr {{.*}} 'void (int *)' Function {{.*}} 'FUnspecified' 'void (int *)'
// CHECK:     `-ImplicitCastExpr {{.*}} 'int *' <BoundsSafetyPointerCast>
// CHECK:       `-ImplicitCastExpr {{.*}} 'int *__single' <LValueToRValue>
// CHECK:         `-DeclRefExpr {{.*}} 'int *__single' lvalue ParmVar {{.*}} 'x' 'int *__single'
void FUnspecifiedInline(int *x) {
  FUnspecified(x);
}

__ptrcheck_abi_assume_unsafe_indexable()

// CHECK-LABEL: FUnsafeIndexable 'void (int *__unsafe_indexable)'
// CHECK: |-ParmVarDecl {{.*}} x 'int *__unsafe_indexable'
// CHECK: `-CompoundStmt
// CHECK:   |-DeclStmt
// CHECK:   | `-VarDecl {{.*}} used y 'int *__unsafe_indexable' cinit
// CHECK:   |   `-ImplicitCastExpr {{.*}} 'int *__unsafe_indexable' <LValueToRValue>
// CHECK:   |     `-DeclRefExpr {{.*}} 'int *__unsafe_indexable' lvalue ParmVar {{.*}} 'x' 'int *__unsafe_indexable'
// CHECK:   |-CompoundAssignOperator {{.*}} 'int *__unsafe_indexable' '-=' ComputeLHSTy='int *__unsafe_indexable' ComputeResultTy='int *__unsafe_indexable'
// CHECK:   | |-DeclRefExpr {{.*}} 'int *__unsafe_indexable' lvalue Var {{.*}} 'y' 'int *__unsafe_indexable'
// CHECK:   | `-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK:   |   `-UnaryOperator {{.*}} 'int' lvalue prefix '*' cannot overflow
// CHECK:   |     `-ImplicitCastExpr {{.*}} 'int *__unsafe_indexable' <LValueToRValue>
// CHECK:   |       `-DeclRefExpr {{.*}} 'int *__unsafe_indexable' lvalue ParmVar {{.*}} 'x' 'int *__unsafe_indexable'
// CHECK:   `-DeclStmt
// CHECK:     `-VarDecl {{.*}} z 'int *__unsafe_indexable*__unsafe_indexable' cinit
// CHECK:       `-UnaryOperator {{.*}} 'int *__unsafe_indexable*__bidi_indexable' prefix '&' cannot overflow
// CHECK:         `-DeclRefExpr {{.*}} 'int *__unsafe_indexable' lvalue Var {{.*}} 'y' 'int *__unsafe_indexable'
void FUnsafeIndexable(int *x) {
  int *y = x;
  y -= *x;
  int **z = &y;
}

// CHECK-LABEL: FUnsafeIndexableAddrOf 'void (void)'
// CHECK: `-CompoundStmt
// CHECK:   |-DeclStmt
// CHECK:   | `-VarDecl {{.*}} used x 'int' cinit
// CHECK:   |   `-IntegerLiteral {{.*}} 'int' 0
// CHECK:   `-DeclStmt
// CHECK:     `-VarDecl {{.*}} y 'int *__bidi_indexable' cinit
// CHECK:       `-UnaryOperator {{.*}} 'int *__bidi_indexable' prefix '&' cannot overflow
// CHECK:         `-DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'x' 'int'
void FUnsafeIndexableAddrOf(void) {
  int x = 0;
  int *__bidi_indexable y = &x;
}

// CHECK-LABEL: FUnsafeIndexableArrayDecay 'void (int *__unsafe_indexable)'
// CHECK: |-ParmVarDecl {{.*}} x 'int *__unsafe_indexable'
// CHECK: `-CompoundStmt
// CHECK:   |-DeclStmt
// CHECK:   | `-VarDecl {{.*}} used arr 'int[2]' cinit
// CHECK:   |   `-InitListExpr {{.*}} 'int[2]'
// CHECK:   |     |-array_filler: ImplicitValueInitExpr {{.*}} 'int'
// CHECK:   |     `-IntegerLiteral {{.*}} 'int' 0
// CHECK:   `-DeclStmt
// CHECK:     `-VarDecl {{.*}} y 'int *__bidi_indexable' cinit
// CHECK:       `-ImplicitCastExpr {{.*}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK:         `-DeclRefExpr {{.*}} 'int[2]' lvalue Var {{.*}} 'arr' 'int[2]'

void FUnsafeIndexableArrayDecay(int *x) {
  int arr[2] = { 0 };
  int *__bidi_indexable y = arr;
}

// CHECK-LABEL: FUnsafeIndexableCountedBy 'void (int *__single __counted_by(len), unsigned int)'
// CHECK:   |-ParmVarDecl [[var_ptr:0x[^ ]+]] {{.+}} ptr
// CHECK:   |-ParmVarDecl [[var_len:0x[^ ]+]] {{.+}} len
// CHECK:   | `-DependerDeclsAttr
// CHECK:   `-CompoundStmt
// CHECK:     `-DeclStmt
// CHECK:       `-VarDecl [[var_y_4:0x[^ ]+]]
// CHECK:         `-ImplicitCastExpr {{.+}} 'int *__unsafe_indexable' <BoundsSafetyPointerCast>
// CHECK:           `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:             |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:             | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK:             | | |-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:             | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK:             | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:             | | | | `-OpaqueValueExpr [[ove]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:             | | | `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'unsigned int'
// CHECK:             | |-OpaqueValueExpr [[ove]]
// CHECK:             | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <LValueToRValue>
// CHECK:             | |   `-DeclRefExpr {{.+}} [[var_ptr]]
// CHECK:             | `-OpaqueValueExpr [[ove_1]]
// CHECK:             |   `-ImplicitCastExpr {{.+}} 'unsigned int' <LValueToRValue>
// CHECK:             |     `-DeclRefExpr {{.+}} [[var_len]]
// CHECK:             |-OpaqueValueExpr [[ove]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:             `-OpaqueValueExpr [[ove_1]] {{.*}} 'unsigned int'
void FUnsafeIndexableCountedBy(int *__counted_by(len) ptr, unsigned len) {
  int *y = ptr;
}
