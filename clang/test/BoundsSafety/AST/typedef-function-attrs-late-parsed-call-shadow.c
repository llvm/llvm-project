
// RUN: %clang_cc1 -fbounds-safety -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

// Make sure that the correct variable (var_c in AST) is used when promoting
// the result of the call f(c) to __bidi_indexable pointer p.

const int count = 16;
// CHECK: VarDecl [[var_count:0x[^ ]+]] {{.*}} count 'const int'
// CHECK: `-IntegerLiteral {{.+}} 16

typedef int *__counted_by(count) cnt_t(int count);
// CHECK: TypedefDecl {{.+}} referenced cnt_t 'int *__single __counted_by(count)(int)'
// CHECK: `-FunctionProtoType {{.+}} 'int *__single __counted_by(count)(int)' cdecl
// CHECK:   |-CountAttributedType {{.+}} 'int *__single __counted_by(count)' sugar
// CHECK:   | `-PointerType {{.+}} 'int *__single'
// CHECK:   |   `-BuiltinType {{.+}} 'int'
// CHECK:   `-BuiltinType {{.+}} 'int'

void foo(cnt_t f, int c) {
  int count = 32;
  int *p = f(c);
}
// CHECK: FunctionDecl [[func_foo:0x[^ ]+]] {{.*}} foo 'void (cnt_t *__single, int)'
// CHECK: |-ParmVarDecl [[var_f:0x[^ ]+]] {{.*}} f 'cnt_t *__single'
// CHECK: |-ParmVarDecl [[var_c:0x[^ ]+]] {{.*}} c 'int'
// CHECK: `-CompoundStmt
// CHECK:   |-DeclStmt
// CHECK:   | `-VarDecl [[var_count_1:0x[^ ]+]] {{.*}} count 'int'
// CHECK:   |   `-IntegerLiteral {{.+}} 32
// CHECK:   `-DeclStmt
// CHECK:     `-VarDecl [[var_p:0x[^ ]+]] {{.*}} p 'int *__bidi_indexable'
// CHECK:       `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:         |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK:         | | |-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:         | | |   `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'int'
// CHECK:         | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK:         | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:         | | | | `-OpaqueValueExpr [[ove]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:         | | | `-OpaqueValueExpr [[ove_1]] {{.*}} 'int'
// CHECK:         | |-OpaqueValueExpr [[ove_1]]
// CHECK:         | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:         | |   `-DeclRefExpr {{.+}} [[var_c]]
// CHECK:         | `-OpaqueValueExpr [[ove]]
// CHECK:         |   `-CallExpr
// CHECK:         |     |-ImplicitCastExpr {{.+}} 'cnt_t *__single' <LValueToRValue>
// CHECK:         |     | `-DeclRefExpr {{.+}} [[var_f]]
// CHECK:         |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'int'
// CHECK:         |-OpaqueValueExpr [[ove_1]] {{.*}} 'int'
// CHECK:         `-OpaqueValueExpr [[ove]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
