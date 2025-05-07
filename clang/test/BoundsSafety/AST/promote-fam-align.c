

// RUN: %clang_cc1 -ast-dump -verify -fbounds-safety %s | FileCheck %s
// RUN: %clang_cc1 -ast-dump -verify -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s | FileCheck %s

// expected-no-diagnostics

#include <ptrcheck.h>
typedef unsigned char uuid_t[16];
struct s {
  int count;
  uuid_t fam[__counted_by(count)];
};

void promote(struct s *info) {
  uuid_t *uuids = &info->fam[0];
  (void)uuids;
}

// CHECK: `-FunctionDecl [[func_foo:0x[^ ]+]] {{.+}} promote
// CHECK:   |-ParmVarDecl [[var_info:0x[^ ]+]]
// CHECK:   `-CompoundStmt
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl [[var_uuids:0x[^ ]+]]
// CHECK:     |   `-UnaryOperator {{.+}} cannot overflow
// CHECK:     |     `-ArraySubscriptExpr
// CHECK:     |       |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     |       | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:     |       | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'uuid_t *__bidi_indexable'
// CHECK:     |       | | | |-ImplicitCastExpr {{.+}} 'uuid_t *' <ArrayToPointerDecay>
// CHECK:     |       | | | | `-MemberExpr {{.+}} ->fam
// CHECK:     |       | | | |   `-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'struct s *__single'
// CHECK:     |       | | | |-GetBoundExpr {{.+}} upper
// CHECK:     |       | | | | `-BoundsSafetyPointerPromotionExpr {{.+}} 'struct s *__bidi_indexable'
// CHECK:     |       | | | |   |-OpaqueValueExpr [[ove]] {{.*}} 'struct s *__single'
// CHECK:     |       | | | |   |-BinaryOperator {{.+}} 'uuid_t *' '+'
// CHECK:     |       | | | |   | |-ImplicitCastExpr {{.+}} 'uuid_t *' <ArrayToPointerDecay>
// CHECK:     |       | | | |   | | `-MemberExpr {{.+}} ->fam
// CHECK:     |       | | | |   | |   `-OpaqueValueExpr [[ove]] {{.*}} 'struct s *__single'
// CHECK:     |       | | | |   | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:     |       | | | |   |   `-MemberExpr {{.+}} ->count
// CHECK:     |       | | | |   |     `-OpaqueValueExpr [[ove]] {{.*}} 'struct s *__single'
// CHECK:     |       | | `-OpaqueValueExpr [[ove]]
// CHECK:     |       | |   `-ImplicitCastExpr {{.+}} 'struct s *__single' <LValueToRValue>
// CHECK:     |       | |     `-DeclRefExpr {{.+}} [[var_info]]
// CHECK:     |       | `-OpaqueValueExpr [[ove]] {{.*}} 'struct s *__single'
// CHECK:     |       `-IntegerLiteral {{.+}} 0
// CHECK:     `-CStyleCastExpr {{.+}} 'void' <ToVoid>
// CHECK:       `-ImplicitCastExpr {{.+}} 'uuid_t *__bidi_indexable' <LValueToRValue>
// CHECK:         `-DeclRefExpr {{.+}} [[var_uuids]]
