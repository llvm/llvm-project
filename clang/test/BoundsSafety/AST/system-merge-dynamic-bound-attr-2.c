

// RUN: %clang_cc1 -ast-dump -fbounds-safety -isystem %S/Inputs/system-merge-dynamic-bound-attr %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -isystem %S/Inputs/system-merge-dynamic-bound-attr %s 2>&1 | FileCheck %s

#include <header-no-attr.h>
#include <header-with-attr.h>

// CHECK: FunctionDecl {{.+}} myalloc
// CHECK: FunctionDecl {{.+}} myalloc
// CHECK: -AllocSizeAttr

void Test(unsigned siz) {
  void *src = myalloc(siz);
}

// CHECK: {{^}}`-FunctionDecl [[func_Test:0x[^ ]+]] {{.+}} Test
// CHECK: {{^}}  |-ParmVarDecl [[var_siz:0x[^ ]+]]
// CHECK: {{^}}  `-CompoundStmt
// CHECK: {{^}}    `-DeclStmt
// CHECK: {{^}}      `-VarDecl [[var_src:0x[^ ]+]]
// CHECK: {{^}}        `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}          |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}          | |-BoundsSafetyPointerPromotionExpr {{.+}} 'void *__bidi_indexable'
// CHECK: {{^}}          | | |-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'void *'
// CHECK: {{^}}          | | |-ImplicitCastExpr {{.+}} 'void *' <BitCast>
// CHECK: {{^}}          | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK: {{^}}          | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK: {{^}}          | | |   | `-OpaqueValueExpr [[ove]] {{.*}} 'void *'
// CHECK: {{^}}          | | |   `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'unsigned int'
// CHECK: {{^}}          | |-OpaqueValueExpr [[ove_1]]
// CHECK: {{^}}          | | `-ImplicitCastExpr {{.+}} 'unsigned int' <LValueToRValue>
// CHECK: {{^}}          | |   `-DeclRefExpr {{.+}} [[var_siz]]
// CHECK: {{^}}          | `-OpaqueValueExpr [[ove]]
// CHECK: {{^}}          |   `-CallExpr
// CHECK: {{^}}          |     |-ImplicitCastExpr {{.+}} 'void *(*__single)(unsigned int)' <FunctionToPointerDecay>
// CHECK: {{^}}          |     | `-DeclRefExpr {{.+}} 'myalloc'
// CHECK: {{^}}          |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'unsigned int'
// CHECK: {{^}}          |-OpaqueValueExpr [[ove_1]] {{.*}} 'unsigned int'
// CHECK: {{^}}          `-OpaqueValueExpr [[ove]] {{.*}} 'void *'
