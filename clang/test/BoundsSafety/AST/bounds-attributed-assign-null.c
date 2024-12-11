// RUN: %clang_cc1 -fbounds-safety -ast-dump -verify %s | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -fbounds-safety-bringup-missing-checks=all -ast-dump -verify %s | FileCheck %s
#include <ptrcheck.h>

// expected-no-diagnostics

// CHECK:      {{^}}|-FunctionDecl [[func_init_local:0x[^ ]+]] {{.+}} init_local
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_count:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |   `-DependerDeclsAttr
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_p:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   `-BoundsCheckExpr {{.+}} 'count == 0'
// CHECK-NEXT: {{^}}|   |     |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   |     | `-CStyleCastExpr {{.+}} 'int *' <NullToPointer>
// CHECK-NEXT: {{^}}|   |     |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |     `-BinaryOperator {{.+}} 'int' '=='
// CHECK-NEXT: {{^}}|   |       |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|   |       | `-DeclRefExpr {{.+}} [[var_count]]
// CHECK-NEXT: {{^}}|   |       `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_count1:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |   `-DependerDeclsAttr
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_p1:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   `-BoundsCheckExpr {{.+}} 'count1 == 0'
// CHECK-NEXT: {{^}}|   |     |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count1)':'int *__single' <NullToPointer>
// CHECK-NEXT: {{^}}|   |     | `-CStyleCastExpr {{.+}} 'void *' <NullToPointer>
// CHECK-NEXT: {{^}}|   |     |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |     `-BinaryOperator {{.+}} 'int' '=='
// CHECK-NEXT: {{^}}|   |       |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|   |       | `-DeclRefExpr {{.+}} [[var_count1]]
// CHECK-NEXT: {{^}}|   |       `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_count2:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |   `-DependerDeclsAttr
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_p2:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   `-BoundsCheckExpr {{.+}} 'count2 == 0'
// CHECK-NEXT: {{^}}|   |     |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count2)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   |     | `-CStyleCastExpr {{.+}} 'int *' <BitCast>
// CHECK-NEXT: {{^}}|   |     |   `-CStyleCastExpr {{.+}} 'void *' <NullToPointer>
// CHECK-NEXT: {{^}}|   |     |     `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |     `-BinaryOperator {{.+}} 'int' '=='
// CHECK-NEXT: {{^}}|   |       |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|   |       | `-DeclRefExpr {{.+}} [[var_count2]]
// CHECK-NEXT: {{^}}|   |       `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_count3:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |   `-DependerDeclsAttr
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_p3:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   `-BoundsCheckExpr {{.+}} 'count3 == 0'
// CHECK-NEXT: {{^}}|   |     |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count3)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   |     | `-ParenExpr
// CHECK-NEXT: {{^}}|   |     |   `-CStyleCastExpr {{.+}} 'int *' <BitCast>
// CHECK-NEXT: {{^}}|   |     |     `-CStyleCastExpr {{.+}} 'void *' <BitCast>
// CHECK-NEXT: {{^}}|   |     |       `-CStyleCastExpr {{.+}} 'int *' <NullToPointer>
// CHECK-NEXT: {{^}}|   |     |         `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |     `-BinaryOperator {{.+}} 'int' '=='
// CHECK-NEXT: {{^}}|   |       |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|   |       | `-DeclRefExpr {{.+}} [[var_count3]]
// CHECK-NEXT: {{^}}|   |       `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_count4:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |   `-DependerDeclsAttr
// CHECK-NEXT: {{^}}|   `-DeclStmt
// CHECK-NEXT: {{^}}|     `-VarDecl [[var_p4:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       `-BoundsCheckExpr {{.+}} 'count4 == 0'
// CHECK-NEXT: {{^}}|         |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count4)':'int *__single' <NullToPointer>
// CHECK-NEXT: {{^}}|         | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|         `-BinaryOperator {{.+}} 'int' '=='
// CHECK-NEXT: {{^}}|           |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|           | `-DeclRefExpr {{.+}} [[var_count4]]
// CHECK-NEXT: {{^}}|           `-IntegerLiteral {{.+}} 0
void init_local(void) {
  int count = 0;
  int* __counted_by(count) p = (int*) 0;

  int count1 = 0;
  int* __counted_by(count1) p1 = (void*) 0;

  int count2 = 0;
  int* __counted_by(count2) p2 = (int*)(void*) 0;

  int count3 = 0;
  int* __counted_by(count3) p3 = ((int*)(void*)(int*) 0);

  int count4 = 0;
  int* __counted_by(count4) p4 = 0;
}

// CHECK: {{^}}|-FunctionDecl [[func_assign_local:0x[^ ]+]] {{.+}} assign_local
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_count_1:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |   `-DependerDeclsAttr
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_p_1:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_count1_1:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |   `-DependerDeclsAttr
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_p1_1:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_count2_1:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |   `-DependerDeclsAttr
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_p2_1:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_count3_1:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |   `-DependerDeclsAttr
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_p3_1:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_count4_1:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |   `-DependerDeclsAttr
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_p4_1:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|   | |-BoundsCheckExpr {{.+}} '0 == 0'
// CHECK-NEXT: {{^}}|   | | |-BinaryOperator {{.+}} 'int' '='
// CHECK-NEXT: {{^}}|   | | | |-DeclRefExpr {{.+}} [[var_count_1]]
// CHECK-NEXT: {{^}}|   | | | `-OpaqueValueExpr [[ove:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   | | |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | | `-BinaryOperator {{.+}} 'int' '=='
// CHECK-NEXT: {{^}}|   | |   |-OpaqueValueExpr [[ove]]
// CHECK-NEXT: {{^}}|   | |   | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | |-OpaqueValueExpr [[ove]]
// CHECK-NEXT: {{^}}|   | | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | `-OpaqueValueExpr [[ove_1:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   `-CStyleCastExpr {{.+}} 'int *' <NullToPointer>
// CHECK-NEXT: {{^}}|   |     `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|   | |-BinaryOperator {{.+}} 'int *__single __counted_by(count)':'int *__single' '='
// CHECK-NEXT: {{^}}|   | | |-DeclRefExpr {{.+}} [[var_p_1]]
// CHECK-NEXT: {{^}}|   | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |   `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *'
// CHECK:      {{^}}|   | |-OpaqueValueExpr [[ove]] {{.*}} 'int'
// CHECK:      {{^}}|   | `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *'
// CHECK:      {{^}}|   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|   | |-BoundsCheckExpr {{.+}} '0 == 0'
// CHECK-NEXT: {{^}}|   | | |-BinaryOperator {{.+}} 'int' '='
// CHECK-NEXT: {{^}}|   | | | |-DeclRefExpr {{.+}} [[var_count1_1]]
// CHECK-NEXT: {{^}}|   | | | `-OpaqueValueExpr [[ove_2:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   | | |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | | `-BinaryOperator {{.+}} 'int' '=='
// CHECK-NEXT: {{^}}|   | |   |-OpaqueValueExpr [[ove_2]]
// CHECK-NEXT: {{^}}|   | |   | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | |-OpaqueValueExpr [[ove_2]]
// CHECK-NEXT: {{^}}|   | | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | `-OpaqueValueExpr [[ove_3:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count1)':'int *__single' <NullToPointer>
// CHECK-NEXT: {{^}}|   |     `-CStyleCastExpr {{.+}} 'void *' <NullToPointer>
// CHECK-NEXT: {{^}}|   |       `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|   | |-BinaryOperator {{.+}} 'int *__single __counted_by(count1)':'int *__single' '='
// CHECK-NEXT: {{^}}|   | | |-DeclRefExpr {{.+}} [[var_p1_1]]
// CHECK-NEXT: {{^}}|   | | `-OpaqueValueExpr [[ove_3]] {{.*}} 'int *__single __counted_by(count1)':'int *__single'
// CHECK:      {{^}}|   | |-OpaqueValueExpr [[ove_2]] {{.*}} 'int'
// CHECK:      {{^}}|   | `-OpaqueValueExpr [[ove_3]] {{.*}} 'int *__single __counted_by(count1)':'int *__single'
// CHECK:      {{^}}|   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|   | |-BoundsCheckExpr {{.+}} '0 == 0'
// CHECK-NEXT: {{^}}|   | | |-BinaryOperator {{.+}} 'int' '='
// CHECK-NEXT: {{^}}|   | | | |-DeclRefExpr {{.+}} [[var_count2_1]]
// CHECK-NEXT: {{^}}|   | | | `-OpaqueValueExpr [[ove_4:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   | | |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | | `-BinaryOperator {{.+}} 'int' '=='
// CHECK-NEXT: {{^}}|   | |   |-OpaqueValueExpr [[ove_4]]
// CHECK-NEXT: {{^}}|   | |   | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | |-OpaqueValueExpr [[ove_4]]
// CHECK-NEXT: {{^}}|   | | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | `-OpaqueValueExpr [[ove_5:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   `-CStyleCastExpr {{.+}} 'int *' <BitCast>
// CHECK-NEXT: {{^}}|   |     `-CStyleCastExpr {{.+}} 'void *' <NullToPointer>
// CHECK-NEXT: {{^}}|   |       `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|   | |-BinaryOperator {{.+}} 'int *__single __counted_by(count2)':'int *__single' '='
// CHECK-NEXT: {{^}}|   | | |-DeclRefExpr {{.+}} [[var_p2_1]]
// CHECK-NEXT: {{^}}|   | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count2)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |   `-OpaqueValueExpr [[ove_5]] {{.*}} 'int *'
// CHECK:      {{^}}|   | |-OpaqueValueExpr [[ove_4]] {{.*}} 'int'
// CHECK:      {{^}}|   | `-OpaqueValueExpr [[ove_5]] {{.*}} 'int *'
// CHECK:      {{^}}|   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|   | |-BoundsCheckExpr {{.+}} '0 == 0'
// CHECK-NEXT: {{^}}|   | | |-BinaryOperator {{.+}} 'int' '='
// CHECK-NEXT: {{^}}|   | | | |-DeclRefExpr {{.+}} [[var_count3_1]]
// CHECK-NEXT: {{^}}|   | | | `-OpaqueValueExpr [[ove_6:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   | | |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | | `-BinaryOperator {{.+}} 'int' '=='
// CHECK-NEXT: {{^}}|   | |   |-OpaqueValueExpr [[ove_6]]
// CHECK-NEXT: {{^}}|   | |   | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | |-OpaqueValueExpr [[ove_6]]
// CHECK-NEXT: {{^}}|   | | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | `-OpaqueValueExpr [[ove_7:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   `-ParenExpr
// CHECK-NEXT: {{^}}|   |     `-CStyleCastExpr {{.+}} 'int *' <BitCast>
// CHECK-NEXT: {{^}}|   |       `-CStyleCastExpr {{.+}} 'void *' <BitCast>
// CHECK-NEXT: {{^}}|   |         `-CStyleCastExpr {{.+}} 'int *' <NullToPointer>
// CHECK-NEXT: {{^}}|   |           `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|   | |-BinaryOperator {{.+}} 'int *__single __counted_by(count3)':'int *__single' '='
// CHECK-NEXT: {{^}}|   | | |-DeclRefExpr {{.+}} [[var_p3_1]]
// CHECK-NEXT: {{^}}|   | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count3)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |   `-OpaqueValueExpr [[ove_7]] {{.*}} 'int *'
// CHECK:      {{^}}|   | |-OpaqueValueExpr [[ove_6]] {{.*}} 'int'
// CHECK:      {{^}}|   | `-OpaqueValueExpr [[ove_7]] {{.*}} 'int *'
// CHECK:      {{^}}|   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|   | |-BoundsCheckExpr {{.+}} '0 == 0'
// CHECK-NEXT: {{^}}|   | | |-BinaryOperator {{.+}} 'int' '='
// CHECK-NEXT: {{^}}|   | | | |-DeclRefExpr {{.+}} [[var_count4_1]]
// CHECK-NEXT: {{^}}|   | | | `-OpaqueValueExpr [[ove_8:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   | | |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | | `-BinaryOperator {{.+}} 'int' '=='
// CHECK-NEXT: {{^}}|   | |   |-OpaqueValueExpr [[ove_8]]
// CHECK-NEXT: {{^}}|   | |   | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | |-OpaqueValueExpr [[ove_8]]
// CHECK-NEXT: {{^}}|   | | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | `-OpaqueValueExpr [[ove_9:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count4)':'int *__single' <NullToPointer>
// CHECK-NEXT: {{^}}|   |     `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|     |-BinaryOperator {{.+}} 'int *__single __counted_by(count4)':'int *__single' '='
// CHECK-NEXT: {{^}}|     | |-DeclRefExpr {{.+}} [[var_p4_1]]
// CHECK-NEXT: {{^}}|     | `-OpaqueValueExpr [[ove_9]] {{.*}} 'int *__single __counted_by(count4)':'int *__single'
// CHECK:      {{^}}|     |-OpaqueValueExpr [[ove_8]] {{.*}} 'int'
// CHECK:      {{^}}|     `-OpaqueValueExpr [[ove_9]] {{.*}} 'int *__single __counted_by(count4)':'int *__single'
// CHECK:      {{^}}|-RecordDecl
// CHECK-NEXT: {{^}}| |-FieldDecl
// CHECK-NEXT: {{^}}| | `-DependerDeclsAttr
// CHECK-NEXT: {{^}}| `-FieldDecl
void assign_local(void) {
  int count = 0;
  int* __counted_by(count) p;
  int count1 = 0;
  int* __counted_by(count1) p1;
  int count2 = 0;
  int* __counted_by(count2) p2;
  int count3 = 0;
  int* __counted_by(count3) p3;
  int count4 = 0;
  int* __counted_by(count4) p4;

  count = 0;
  p = (int*)0;

  count1 = 0;
  p1 = (void*) 0;

  count2 = 0;
  p2 = (int*)(void*) 0;

  count3 = 0;
  p3 = ((int*)(void*)(int*) 0);

  count4 = 0;
  p4 = 0;
}

struct Cb {
  int count;
  int* __counted_by(count) ptr;
};

// CHECK: {{^}}|-FunctionDecl [[func_init_struct:0x[^ ]+]] {{.+}} init_struct
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_c:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   `-BoundsCheckExpr {{.+}} '0 == 0'
// CHECK-NEXT: {{^}}|   |     |-InitListExpr
// CHECK-NEXT: {{^}}|   |     | |-OpaqueValueExpr [[ove_10:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |     | | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |     | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   |     |   `-OpaqueValueExpr [[ove_11:0x[^ ]+]] {{.*}} 'int *'
// CHECK:      {{^}}|   |     |-BinaryOperator {{.+}} 'int' '=='
// CHECK-NEXT: {{^}}|   |     | |-OpaqueValueExpr [[ove_10]]
// CHECK-NEXT: {{^}}|   |     | | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |     | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |     |-OpaqueValueExpr [[ove_10]]
// CHECK-NEXT: {{^}}|   |     | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |     `-OpaqueValueExpr [[ove_11]]
// CHECK-NEXT: {{^}}|   |       `-CStyleCastExpr {{.+}} 'int *' <NullToPointer>
// CHECK-NEXT: {{^}}|   |         `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_c1:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   `-BoundsCheckExpr {{.+}} '0 == 0'
// CHECK-NEXT: {{^}}|   |     |-InitListExpr
// CHECK-NEXT: {{^}}|   |     | |-OpaqueValueExpr [[ove_12:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |     | | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |     | `-OpaqueValueExpr [[ove_13:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |     |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)':'int *__single' <NullToPointer>
// CHECK-NEXT: {{^}}|   |     |     `-CStyleCastExpr {{.+}} 'void *' <NullToPointer>
// CHECK-NEXT: {{^}}|   |     |       `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |     |-BinaryOperator {{.+}} 'int' '=='
// CHECK-NEXT: {{^}}|   |     | |-OpaqueValueExpr [[ove_12]]
// CHECK-NEXT: {{^}}|   |     | | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |     | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |     |-OpaqueValueExpr [[ove_12]]
// CHECK-NEXT: {{^}}|   |     | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |     `-OpaqueValueExpr [[ove_13]]
// CHECK-NEXT: {{^}}|   |       `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)':'int *__single' <NullToPointer>
// CHECK-NEXT: {{^}}|   |         `-CStyleCastExpr {{.+}} 'void *' <NullToPointer>
// CHECK-NEXT: {{^}}|   |           `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_c2:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   `-BoundsCheckExpr {{.+}} '0 == 0'
// CHECK-NEXT: {{^}}|   |     |-InitListExpr
// CHECK-NEXT: {{^}}|   |     | |-OpaqueValueExpr [[ove_14:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |     | | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |     | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   |     |   `-OpaqueValueExpr [[ove_15:0x[^ ]+]] {{.*}} 'int *'
// CHECK:      {{^}}|   |     |-BinaryOperator {{.+}} 'int' '=='
// CHECK-NEXT: {{^}}|   |     | |-OpaqueValueExpr [[ove_14]]
// CHECK-NEXT: {{^}}|   |     | | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |     | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |     |-OpaqueValueExpr [[ove_14]]
// CHECK-NEXT: {{^}}|   |     | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |     `-OpaqueValueExpr [[ove_15]]
// CHECK-NEXT: {{^}}|   |       `-CStyleCastExpr {{.+}} 'int *' <BitCast>
// CHECK-NEXT: {{^}}|   |         `-CStyleCastExpr {{.+}} 'void *' <NullToPointer>
// CHECK-NEXT: {{^}}|   |           `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_c3:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   `-BoundsCheckExpr {{.+}} '0 == 0'
// CHECK-NEXT: {{^}}|   |     |-InitListExpr
// CHECK-NEXT: {{^}}|   |     | |-OpaqueValueExpr [[ove_16:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |     | | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |     | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   |     |   `-OpaqueValueExpr [[ove_17:0x[^ ]+]] {{.*}} 'int *'
// CHECK:      {{^}}|   |     |-BinaryOperator {{.+}} 'int' '=='
// CHECK-NEXT: {{^}}|   |     | |-OpaqueValueExpr [[ove_16]]
// CHECK-NEXT: {{^}}|   |     | | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |     | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |     |-OpaqueValueExpr [[ove_16]]
// CHECK-NEXT: {{^}}|   |     | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |     `-OpaqueValueExpr [[ove_17]]
// CHECK-NEXT: {{^}}|   |       `-ParenExpr
// CHECK-NEXT: {{^}}|   |         `-CStyleCastExpr {{.+}} 'int *' <BitCast>
// CHECK-NEXT: {{^}}|   |           `-CStyleCastExpr {{.+}} 'void *' <BitCast>
// CHECK-NEXT: {{^}}|   |             `-CStyleCastExpr {{.+}} 'int *' <NullToPointer>
// CHECK-NEXT: {{^}}|   |               `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   `-DeclStmt
// CHECK-NEXT: {{^}}|     `-VarDecl [[var_c4:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       `-BoundsCheckExpr {{.+}} '0 == 0'
// CHECK-NEXT: {{^}}|         |-InitListExpr
// CHECK-NEXT: {{^}}|         | |-OpaqueValueExpr [[ove_18:0x[^ ]+]]
// CHECK-NEXT: {{^}}|         | | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|         | `-OpaqueValueExpr [[ove_19:0x[^ ]+]]
// CHECK-NEXT: {{^}}|         |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)':'int *__single' <NullToPointer>
// CHECK-NEXT: {{^}}|         |     `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|         |-BinaryOperator {{.+}} 'int' '=='
// CHECK-NEXT: {{^}}|         | |-OpaqueValueExpr [[ove_18]]
// CHECK-NEXT: {{^}}|         | | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|         | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|         |-OpaqueValueExpr [[ove_18]]
// CHECK-NEXT: {{^}}|         | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|         `-OpaqueValueExpr [[ove_19]]
// CHECK-NEXT: {{^}}|           `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)':'int *__single' <NullToPointer>
// CHECK-NEXT: {{^}}|             `-IntegerLiteral {{.+}} 0
void init_struct(void) {
  struct Cb c = { 0, (int*)0};
  struct Cb c1 = { 0, (void*)0};
  struct Cb c2 = { 0, (int*)(void*)0};
  struct Cb c3 = { 0, ((int*)(void*)(int*)0)};
  struct Cb c4 = { 0, 0};
}

// CHECK: {{^}}|-FunctionDecl [[func_assign_struct:0x[^ ]+]] {{.+}} assign_struct
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_c_1:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_c1_1:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_c2_1:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_c3_1:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_c4_1:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|   | |-BoundsCheckExpr {{.+}} '0 == 0'
// CHECK-NEXT: {{^}}|   | | |-BinaryOperator {{.+}} 'int' '='
// CHECK-NEXT: {{^}}|   | | | |-MemberExpr {{.+}} .count
// CHECK-NEXT: {{^}}|   | | | | `-DeclRefExpr {{.+}} [[var_c_1]]
// CHECK-NEXT: {{^}}|   | | | `-OpaqueValueExpr [[ove_20:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   | | |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | | `-BinaryOperator {{.+}} 'int' '=='
// CHECK-NEXT: {{^}}|   | |   |-OpaqueValueExpr [[ove_20]]
// CHECK-NEXT: {{^}}|   | |   | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | |-OpaqueValueExpr [[ove_20]]
// CHECK-NEXT: {{^}}|   | | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | `-OpaqueValueExpr [[ove_21:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   `-CStyleCastExpr {{.+}} 'int *' <NullToPointer>
// CHECK-NEXT: {{^}}|   |     `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|   | |-BinaryOperator {{.+}} 'int *__single __counted_by(count)':'int *__single' '='
// CHECK-NEXT: {{^}}|   | | |-MemberExpr {{.+}} .ptr
// CHECK-NEXT: {{^}}|   | | | `-DeclRefExpr {{.+}} [[var_c_1]]
// CHECK-NEXT: {{^}}|   | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |   `-OpaqueValueExpr [[ove_21]] {{.*}} 'int *'
// CHECK:      {{^}}|   | |-OpaqueValueExpr [[ove_20]] {{.*}} 'int'
// CHECK:      {{^}}|   | `-OpaqueValueExpr [[ove_21]] {{.*}} 'int *'
// CHECK:      {{^}}|   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|   | |-BoundsCheckExpr {{.+}} '0 == 0'
// CHECK-NEXT: {{^}}|   | | |-BinaryOperator {{.+}} 'int' '='
// CHECK-NEXT: {{^}}|   | | | |-MemberExpr {{.+}} .count
// CHECK-NEXT: {{^}}|   | | | | `-DeclRefExpr {{.+}} [[var_c1_1]]
// CHECK-NEXT: {{^}}|   | | | `-OpaqueValueExpr [[ove_22:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   | | |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | | `-BinaryOperator {{.+}} 'int' '=='
// CHECK-NEXT: {{^}}|   | |   |-OpaqueValueExpr [[ove_22]]
// CHECK-NEXT: {{^}}|   | |   | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | |-OpaqueValueExpr [[ove_22]]
// CHECK-NEXT: {{^}}|   | | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | `-OpaqueValueExpr [[ove_23:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)':'int *__single' <NullToPointer>
// CHECK-NEXT: {{^}}|   |     `-CStyleCastExpr {{.+}} 'void *' <NullToPointer>
// CHECK-NEXT: {{^}}|   |       `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|   | |-BinaryOperator {{.+}} 'int *__single __counted_by(count)':'int *__single' '='
// CHECK-NEXT: {{^}}|   | | |-MemberExpr {{.+}} .ptr
// CHECK-NEXT: {{^}}|   | | | `-DeclRefExpr {{.+}} [[var_c1_1]]
// CHECK-NEXT: {{^}}|   | | `-OpaqueValueExpr [[ove_23]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:      {{^}}|   | |-OpaqueValueExpr [[ove_22]] {{.*}} 'int'
// CHECK:      {{^}}|   | `-OpaqueValueExpr [[ove_23]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:      {{^}}|   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|   | |-BoundsCheckExpr {{.+}} '0 == 0'
// CHECK-NEXT: {{^}}|   | | |-BinaryOperator {{.+}} 'int' '='
// CHECK-NEXT: {{^}}|   | | | |-MemberExpr {{.+}} .count
// CHECK-NEXT: {{^}}|   | | | | `-DeclRefExpr {{.+}} [[var_c2_1]]
// CHECK-NEXT: {{^}}|   | | | `-OpaqueValueExpr [[ove_24:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   | | |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | | `-BinaryOperator {{.+}} 'int' '=='
// CHECK-NEXT: {{^}}|   | |   |-OpaqueValueExpr [[ove_24]]
// CHECK-NEXT: {{^}}|   | |   | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | |-OpaqueValueExpr [[ove_24]]
// CHECK-NEXT: {{^}}|   | | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | `-OpaqueValueExpr [[ove_25:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   `-CStyleCastExpr {{.+}} 'int *' <BitCast>
// CHECK-NEXT: {{^}}|   |     `-CStyleCastExpr {{.+}} 'void *' <NullToPointer>
// CHECK-NEXT: {{^}}|   |       `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|   | |-BinaryOperator {{.+}} 'int *__single __counted_by(count)':'int *__single' '='
// CHECK-NEXT: {{^}}|   | | |-MemberExpr {{.+}} .ptr
// CHECK-NEXT: {{^}}|   | | | `-DeclRefExpr {{.+}} [[var_c2_1]]
// CHECK-NEXT: {{^}}|   | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |   `-OpaqueValueExpr [[ove_25]] {{.*}} 'int *'
// CHECK:      {{^}}|   | |-OpaqueValueExpr [[ove_24]] {{.*}} 'int'
// CHECK:      {{^}}|   | `-OpaqueValueExpr [[ove_25]] {{.*}} 'int *'
// CHECK:      {{^}}|   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|   | |-BoundsCheckExpr {{.+}} '0 == 0'
// CHECK-NEXT: {{^}}|   | | |-BinaryOperator {{.+}} 'int' '='
// CHECK-NEXT: {{^}}|   | | | |-MemberExpr {{.+}} .count
// CHECK-NEXT: {{^}}|   | | | | `-DeclRefExpr {{.+}} [[var_c3_1]]
// CHECK-NEXT: {{^}}|   | | | `-OpaqueValueExpr [[ove_26:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   | | |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | | `-BinaryOperator {{.+}} 'int' '=='
// CHECK-NEXT: {{^}}|   | |   |-OpaqueValueExpr [[ove_26]]
// CHECK-NEXT: {{^}}|   | |   | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | |-OpaqueValueExpr [[ove_26]]
// CHECK-NEXT: {{^}}|   | | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | `-OpaqueValueExpr [[ove_27:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   `-ParenExpr
// CHECK-NEXT: {{^}}|   |     `-CStyleCastExpr {{.+}} 'int *' <BitCast>
// CHECK-NEXT: {{^}}|   |       `-CStyleCastExpr {{.+}} 'void *' <BitCast>
// CHECK-NEXT: {{^}}|   |         `-CStyleCastExpr {{.+}} 'int *' <NullToPointer>
// CHECK-NEXT: {{^}}|   |           `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|   | |-BinaryOperator {{.+}} 'int *__single __counted_by(count)':'int *__single' '='
// CHECK-NEXT: {{^}}|   | | |-MemberExpr {{.+}} .ptr
// CHECK-NEXT: {{^}}|   | | | `-DeclRefExpr {{.+}} [[var_c3_1]]
// CHECK-NEXT: {{^}}|   | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |   `-OpaqueValueExpr [[ove_27]] {{.*}} 'int *'
// CHECK:      {{^}}|   | |-OpaqueValueExpr [[ove_26]] {{.*}} 'int'
// CHECK:      {{^}}|   | `-OpaqueValueExpr [[ove_27]] {{.*}} 'int *'
// CHECK:      {{^}}|   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|   | |-BoundsCheckExpr {{.+}} '0 == 0'
// CHECK-NEXT: {{^}}|   | | |-BinaryOperator {{.+}} 'int' '='
// CHECK-NEXT: {{^}}|   | | | |-MemberExpr {{.+}} .count
// CHECK-NEXT: {{^}}|   | | | | `-DeclRefExpr {{.+}} [[var_c4_1]]
// CHECK-NEXT: {{^}}|   | | | `-OpaqueValueExpr [[ove_28:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   | | |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | | `-BinaryOperator {{.+}} 'int' '=='
// CHECK-NEXT: {{^}}|   | |   |-OpaqueValueExpr [[ove_28]]
// CHECK-NEXT: {{^}}|   | |   | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | |-OpaqueValueExpr [[ove_28]]
// CHECK-NEXT: {{^}}|   | | `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | `-OpaqueValueExpr [[ove_29:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)':'int *__single' <NullToPointer>
// CHECK-NEXT: {{^}}|   |     `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|     |-BinaryOperator {{.+}} 'int *__single __counted_by(count)':'int *__single' '='
// CHECK-NEXT: {{^}}|     | |-MemberExpr {{.+}} .ptr
// CHECK-NEXT: {{^}}|     | | `-DeclRefExpr {{.+}} [[var_c4_1]]
// CHECK-NEXT: {{^}}|     | `-OpaqueValueExpr [[ove_29]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:      {{^}}|     |-OpaqueValueExpr [[ove_28]] {{.*}} 'int'
// CHECK:      {{^}}|     `-OpaqueValueExpr [[ove_29]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
void assign_struct(void) {
  struct Cb c; 
  struct Cb c1; 
  struct Cb c2;
  struct Cb c3;
  struct Cb c4;

  c.count = 0;
  c.ptr = (int*)0;

  c1.count = 0;
  c1.ptr = (void*)0;

  c2.count = 0;
  c2.ptr = (int*)(void*)0;

  c3.count = 0;
  c3.ptr = ((int*)(void*)(int*)0);

  c4.count = 0;
  c4.ptr = 0;
}

// CHECK:      {{^}}|-FunctionDecl [[func_receive_cb:0x[^ ]+]] {{.+}} receive_cb
// CHECK-NEXT: {{^}}| |-ParmVarDecl
// CHECK-NEXT: {{^}}| `-ParmVarDecl [[var_count_2:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   `-DependerDeclsAttr
void receive_cb(int* __counted_by(count), int count);

// CHECK: {{^}}`-FunctionDecl [[func_call_arg:0x[^ ]+]] {{.+}} call_arg
// CHECK-NEXT: {{^}}  |-ParmVarDecl [[var_count_3:0x[^ ]+]]
// CHECK-NEXT: {{^}}  `-CompoundStmt
// CHECK-NEXT: {{^}}    |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}    | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}    | | |-BoundsCheckExpr {{.+}} '(int *)0 <= (int *)0 && (int *)0 <= (int *)0 && count <= (int *)0 - (int *)0 && 0 <= count'
// CHECK-NEXT: {{^}}    | | | |-CallExpr
// CHECK-NEXT: {{^}}    | | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *__single __counted_by(count), int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}    | | | | | `-DeclRefExpr {{.+}} [[func_receive_cb]]
// CHECK-NEXT: {{^}}    | | | | |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}    | | | | | `-OpaqueValueExpr [[ove_30:0x[^ ]+]] {{.*}} 'int *'
// CHECK:      {{^}}    | | | | `-OpaqueValueExpr [[ove_31:0x[^ ]+]]
// CHECK-NEXT: {{^}}    | | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}    | | | |     `-DeclRefExpr {{.+}} [[var_count_3]]
// CHECK-NEXT: {{^}}    | | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}    | | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}    | | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}    | | |   | | |-OpaqueValueExpr [[ove_30]] {{.*}} 'int *'
// CHECK:      {{^}}    | | |   | | `-OpaqueValueExpr [[ove_30]] {{.*}} 'int *'
// CHECK:      {{^}}    | | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}    | | |   |   |-OpaqueValueExpr [[ove_30]] {{.*}} 'int *'
// CHECK:      {{^}}    | | |   |   `-OpaqueValueExpr [[ove_30]] {{.*}} 'int *'
// CHECK:      {{^}}    | | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}    | | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}    | | |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: {{^}}    | | |     | | `-OpaqueValueExpr [[ove_31]] {{.*}} 'int'
// CHECK:      {{^}}    | | |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: {{^}}    | | |     |   |-OpaqueValueExpr [[ove_30]] {{.*}} 'int *'
// CHECK:      {{^}}    | | |     |   `-OpaqueValueExpr [[ove_30]] {{.*}} 'int *'
// CHECK:      {{^}}    | | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}    | | |       |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}    | | |       `-OpaqueValueExpr [[ove_31]] {{.*}} 'int'
// CHECK:      {{^}}    | | |-OpaqueValueExpr [[ove_30]]
// CHECK-NEXT: {{^}}    | | | `-CStyleCastExpr {{.+}} 'int *' <NullToPointer>
// CHECK-NEXT: {{^}}    | | |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}    | | `-OpaqueValueExpr [[ove_31]]
// CHECK-NEXT: {{^}}    | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}    | |     `-DeclRefExpr {{.+}} [[var_count_3]]
// CHECK-NEXT: {{^}}    | |-OpaqueValueExpr [[ove_30]] {{.*}} 'int *'
// CHECK:      {{^}}    | `-OpaqueValueExpr [[ove_31]] {{.*}} 'int'
// CHECK:      {{^}}    |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}    | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}    | | |-BoundsCheckExpr {{.+}} '(void *)0 <= (void *)0 && (void *)0 <= (void *)0 && count <= (void *)0 - (void *)0 && 0 <= count'
// CHECK-NEXT: {{^}}    | | | |-CallExpr
// CHECK-NEXT: {{^}}    | | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *__single __counted_by(count), int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}    | | | | | `-DeclRefExpr {{.+}} [[func_receive_cb]]
// CHECK-NEXT: {{^}}    | | | | |-OpaqueValueExpr [[ove_32:0x[^ ]+]]
// CHECK-NEXT: {{^}}    | | | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)':'int *__single' <NullToPointer>
// CHECK-NEXT: {{^}}    | | | | |   `-CStyleCastExpr {{.+}} 'void *' <NullToPointer>
// CHECK-NEXT: {{^}}    | | | | |     `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}    | | | | `-OpaqueValueExpr [[ove_33:0x[^ ]+]]
// CHECK-NEXT: {{^}}    | | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}    | | | |     `-DeclRefExpr {{.+}} [[var_count_3]]
// CHECK-NEXT: {{^}}    | | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}    | | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}    | | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}    | | |   | | |-OpaqueValueExpr [[ove_32]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:      {{^}}    | | |   | | `-OpaqueValueExpr [[ove_32]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:      {{^}}    | | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}    | | |   |   |-OpaqueValueExpr [[ove_32]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:      {{^}}    | | |   |   `-OpaqueValueExpr [[ove_32]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:      {{^}}    | | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}    | | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}    | | |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: {{^}}    | | |     | | `-OpaqueValueExpr [[ove_33]] {{.*}} 'int'
// CHECK:      {{^}}    | | |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: {{^}}    | | |     |   |-OpaqueValueExpr [[ove_32]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:      {{^}}    | | |     |   `-OpaqueValueExpr [[ove_32]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:      {{^}}    | | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}    | | |       |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}    | | |       `-OpaqueValueExpr [[ove_33]] {{.*}} 'int'
// CHECK:      {{^}}    | | |-OpaqueValueExpr [[ove_32]]
// CHECK-NEXT: {{^}}    | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)':'int *__single' <NullToPointer>
// CHECK-NEXT: {{^}}    | | |   `-CStyleCastExpr {{.+}} 'void *' <NullToPointer>
// CHECK-NEXT: {{^}}    | | |     `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}    | | `-OpaqueValueExpr [[ove_33]]
// CHECK-NEXT: {{^}}    | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}    | |     `-DeclRefExpr {{.+}} [[var_count_3]]
// CHECK-NEXT: {{^}}    | |-OpaqueValueExpr [[ove_32]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:      {{^}}    | `-OpaqueValueExpr [[ove_33]] {{.*}} 'int'
// CHECK:      {{^}}    |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}    | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}    | | |-BoundsCheckExpr {{.+}} '(int *)(void *)0 <= (int *)(void *)0 && (int *)(void *)0 <= (int *)(void *)0 && count <= (int *)(void *)0 - (int *)(void *)0 && 0 <= count'
// CHECK-NEXT: {{^}}    | | | |-CallExpr
// CHECK-NEXT: {{^}}    | | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *__single __counted_by(count), int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}    | | | | | `-DeclRefExpr {{.+}} [[func_receive_cb]]
// CHECK-NEXT: {{^}}    | | | | |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}    | | | | | `-OpaqueValueExpr [[ove_34:0x[^ ]+]] {{.*}} 'int *'
// CHECK:      {{^}}    | | | | `-OpaqueValueExpr [[ove_35:0x[^ ]+]]
// CHECK-NEXT: {{^}}    | | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}    | | | |     `-DeclRefExpr {{.+}} [[var_count_3]]
// CHECK-NEXT: {{^}}    | | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}    | | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}    | | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}    | | |   | | |-OpaqueValueExpr [[ove_34]] {{.*}} 'int *'
// CHECK:      {{^}}    | | |   | | `-OpaqueValueExpr [[ove_34]] {{.*}} 'int *'
// CHECK:      {{^}}    | | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}    | | |   |   |-OpaqueValueExpr [[ove_34]] {{.*}} 'int *'
// CHECK:      {{^}}    | | |   |   `-OpaqueValueExpr [[ove_34]] {{.*}} 'int *'
// CHECK:      {{^}}    | | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}    | | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}    | | |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: {{^}}    | | |     | | `-OpaqueValueExpr [[ove_35]] {{.*}} 'int'
// CHECK:      {{^}}    | | |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: {{^}}    | | |     |   |-OpaqueValueExpr [[ove_34]] {{.*}} 'int *'
// CHECK:      {{^}}    | | |     |   `-OpaqueValueExpr [[ove_34]] {{.*}} 'int *'
// CHECK:      {{^}}    | | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}    | | |       |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}    | | |       `-OpaqueValueExpr [[ove_35]] {{.*}} 'int'
// CHECK:      {{^}}    | | |-OpaqueValueExpr [[ove_34]]
// CHECK-NEXT: {{^}}    | | | `-CStyleCastExpr {{.+}} 'int *' <BitCast>
// CHECK-NEXT: {{^}}    | | |   `-CStyleCastExpr {{.+}} 'void *' <NullToPointer>
// CHECK-NEXT: {{^}}    | | |     `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}    | | `-OpaqueValueExpr [[ove_35]]
// CHECK-NEXT: {{^}}    | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}    | |     `-DeclRefExpr {{.+}} [[var_count_3]]
// CHECK-NEXT: {{^}}    | |-OpaqueValueExpr [[ove_34]] {{.*}} 'int *'
// CHECK:      {{^}}    | `-OpaqueValueExpr [[ove_35]] {{.*}} 'int'
// CHECK:      {{^}}    |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}    | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}    | | |-BoundsCheckExpr {{.+}} '((int *)(void *)(int *)0) <= ((int *)(void *)(int *)0) && ((int *)(void *)(int *)0) <= ((int *)(void *)(int *)0) && count <= ((int *)(void *)(int *)0) - ((int *)(void *)(int *)0) && 0 <= count'
// CHECK-NEXT: {{^}}    | | | |-CallExpr
// CHECK-NEXT: {{^}}    | | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *__single __counted_by(count), int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}    | | | | | `-DeclRefExpr {{.+}} [[func_receive_cb]]
// CHECK-NEXT: {{^}}    | | | | |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}    | | | | | `-OpaqueValueExpr [[ove_36:0x[^ ]+]] {{.*}} 'int *'
// CHECK:      {{^}}    | | | | `-OpaqueValueExpr [[ove_37:0x[^ ]+]]
// CHECK-NEXT: {{^}}    | | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}    | | | |     `-DeclRefExpr {{.+}} [[var_count_3]]
// CHECK-NEXT: {{^}}    | | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}    | | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}    | | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}    | | |   | | |-OpaqueValueExpr [[ove_36]] {{.*}} 'int *'
// CHECK:      {{^}}    | | |   | | `-OpaqueValueExpr [[ove_36]] {{.*}} 'int *'
// CHECK:      {{^}}    | | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}    | | |   |   |-OpaqueValueExpr [[ove_36]] {{.*}} 'int *'
// CHECK:      {{^}}    | | |   |   `-OpaqueValueExpr [[ove_36]] {{.*}} 'int *'
// CHECK:      {{^}}    | | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}    | | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}    | | |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: {{^}}    | | |     | | `-OpaqueValueExpr [[ove_37]] {{.*}} 'int'
// CHECK:      {{^}}    | | |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: {{^}}    | | |     |   |-OpaqueValueExpr [[ove_36]] {{.*}} 'int *'
// CHECK:      {{^}}    | | |     |   `-OpaqueValueExpr [[ove_36]] {{.*}} 'int *'
// CHECK:      {{^}}    | | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}    | | |       |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}    | | |       `-OpaqueValueExpr [[ove_37]] {{.*}} 'int'
// CHECK:      {{^}}    | | |-OpaqueValueExpr [[ove_36]]
// CHECK-NEXT: {{^}}    | | | `-ParenExpr
// CHECK-NEXT: {{^}}    | | |   `-CStyleCastExpr {{.+}} 'int *' <BitCast>
// CHECK-NEXT: {{^}}    | | |     `-CStyleCastExpr {{.+}} 'void *' <BitCast>
// CHECK-NEXT: {{^}}    | | |       `-CStyleCastExpr {{.+}} 'int *' <NullToPointer>
// CHECK-NEXT: {{^}}    | | |         `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}    | | `-OpaqueValueExpr [[ove_37]]
// CHECK-NEXT: {{^}}    | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}    | |     `-DeclRefExpr {{.+}} [[var_count_3]]
// CHECK-NEXT: {{^}}    | |-OpaqueValueExpr [[ove_36]] {{.*}} 'int *'
// CHECK:      {{^}}    | `-OpaqueValueExpr [[ove_37]] {{.*}} 'int'
// CHECK:      {{^}}    `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}      |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}      | |-BoundsCheckExpr {{.+}} '0 <= 0 && 0 <= 0 && count <= 0 - 0 && 0 <= count'
// CHECK-NEXT: {{^}}      | | |-CallExpr
// CHECK-NEXT: {{^}}      | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *__single __counted_by(count), int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}      | | | | `-DeclRefExpr {{.+}} [[func_receive_cb]]
// CHECK-NEXT: {{^}}      | | | |-OpaqueValueExpr [[ove_38:0x[^ ]+]]
// CHECK-NEXT: {{^}}      | | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)':'int *__single' <NullToPointer>
// CHECK-NEXT: {{^}}      | | | |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}      | | | `-OpaqueValueExpr [[ove_39:0x[^ ]+]]
// CHECK-NEXT: {{^}}      | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}      | | |     `-DeclRefExpr {{.+}} [[var_count_3]]
// CHECK-NEXT: {{^}}      | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}      | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}      | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}      | |   | | |-OpaqueValueExpr [[ove_38]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:      {{^}}      | |   | | `-OpaqueValueExpr [[ove_38]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:      {{^}}      | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}      | |   |   |-OpaqueValueExpr [[ove_38]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:      {{^}}      | |   |   `-OpaqueValueExpr [[ove_38]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:      {{^}}      | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}      | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}      | |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: {{^}}      | |     | | `-OpaqueValueExpr [[ove_39]] {{.*}} 'int'
// CHECK:      {{^}}      | |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: {{^}}      | |     |   |-OpaqueValueExpr [[ove_38]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:      {{^}}      | |     |   `-OpaqueValueExpr [[ove_38]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:      {{^}}      | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}      | |       |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}      | |       `-OpaqueValueExpr [[ove_39]] {{.*}} 'int'
// CHECK:      {{^}}      | |-OpaqueValueExpr [[ove_38]]
// CHECK-NEXT: {{^}}      | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)':'int *__single' <NullToPointer>
// CHECK-NEXT: {{^}}      | |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}      | `-OpaqueValueExpr [[ove_39]]
// CHECK-NEXT: {{^}}      |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}      |     `-DeclRefExpr {{.+}} [[var_count_3]]
// CHECK-NEXT: {{^}}      |-OpaqueValueExpr [[ove_38]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:      {{^}}      `-OpaqueValueExpr [[ove_39]] {{.*}} 'int'
void call_arg(int count) {
  receive_cb((int*)0, count);
  receive_cb((void*)0, count);
  receive_cb((int*)(void*)0, count);
  receive_cb(((int*)(void*)(int*)0), count);
  receive_cb(0, count);
}

// Assignment on returns and in compound-literals are handled elsewhere
