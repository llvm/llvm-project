// RUN: %clang_cc1 -triple aarch64 -target-feature +mte -ast-dump %s | FileCheck %s

#include <stddef.h>
#include <arm_acle.h>

struct A;
struct A *ptr_a;
struct A *ptr_a2;
unsigned int uval;
ptrdiff_t pdval;

void test() {
  // CHECK:      |-FunctionDecl {{.+}} test 'void ()'
  // CHECK-NEXT: | `-CompoundStmt

  ptr_a = __arm_mte_create_random_tag(ptr_a, uval);
  // CHECK-NEXT: |   |-BinaryOperator {{.+}} 'struct A *' '='
  // CHECK-NEXT: |   | |-DeclRefExpr {{.+}} 'struct A *' lvalue Var {{.+}} 'ptr_a' 'struct A *'
  // CHECK-NEXT: |   | `-CallExpr {{.+}} 'struct A *'
  // CHECK-NEXT: |   |   |-ImplicitCastExpr {{.+}} 'void *(*)(void *, unsigned long)' <BuiltinFnToFnPtr>
  // CHECK-NEXT: |   |   | `-DeclRefExpr {{.+}} '<builtin fn type>' Function {{.+}} '__builtin_arm_irg' 'void *(void *, unsigned long)'
  // CHECK-NEXT: |   |   |-ImplicitCastExpr {{.+}} 'struct A *' <LValueToRValue>
  // CHECK-NEXT: |   |   | `-DeclRefExpr {{.+}} 'struct A *' lvalue Var {{.+}} 'ptr_a' 'struct A *'
  // CHECK-NEXT: |   |   `-ImplicitCastExpr {{.+}} 'unsigned int' <LValueToRValue>
  // CHECK-NEXT: |   |     `-DeclRefExpr {{.+}} 'unsigned int' lvalue Var {{.+}} 'uval' 'unsigned int'

  ptr_a = __arm_mte_increment_tag(ptr_a, 5);
  // CHECK-NEXT: |   |-BinaryOperator {{.+}} 'struct A *' '='
  // CHECK-NEXT: |   | |-DeclRefExpr {{.+}} 'struct A *' lvalue Var {{.+}} 'ptr_a' 'struct A *'
  // CHECK-NEXT: |   | `-CallExpr {{.+}} 'struct A *'
  // CHECK-NEXT: |   |   |-ImplicitCastExpr {{.+}} 'void *(*)(void *, unsigned int)' <BuiltinFnToFnPtr>
  // CHECK-NEXT: |   |   | `-DeclRefExpr {{.+}} '<builtin fn type>' Function {{.+}} '__builtin_arm_addg' 'void *(void *, unsigned int)'
  // CHECK-NEXT: |   |   |-ImplicitCastExpr {{.+}} 'struct A *' <LValueToRValue>
  // CHECK-NEXT: |   |   | `-DeclRefExpr {{.+}} 'struct A *' lvalue Var {{.+}} 'ptr_a' 'struct A *'
  // CHECK-NEXT: |   |   `-IntegerLiteral {{.+}} 'int' 5

  uval = __arm_mte_exclude_tag(ptr_a, uval);
  // CHECK-NEXT: |   |-BinaryOperator {{.+}} 'unsigned int' '='
  // CHECK-NEXT: |   | |-DeclRefExpr {{.+}} 'unsigned int' lvalue Var {{.+}} 'uval' 'unsigned int'
  // CHECK-NEXT: |   | `-ImplicitCastExpr {{.+}} 'unsigned int' <IntegralCast>
  // CHECK-NEXT: |   |   `-CallExpr {{.+}} 'unsigned long'
  // CHECK-NEXT: |   |     |-ImplicitCastExpr {{.+}} 'unsigned long (*)(void *, unsigned long)' <BuiltinFnToFnPtr>
  // CHECK-NEXT: |   |     | `-DeclRefExpr {{.+}} '<builtin fn type>' Function {{.+}} '__builtin_arm_gmi' 'unsigned long (void *, unsigned long)'
  // CHECK-NEXT: |   |     |-ImplicitCastExpr {{.+}} 'struct A *' <LValueToRValue>
  // CHECK-NEXT: |   |     | `-DeclRefExpr {{.+}} 'struct A *' lvalue Var {{.+}} 'ptr_a' 'struct A *'
  // CHECK-NEXT: |   |     `-ImplicitCastExpr {{.+}} 'unsigned int' <LValueToRValue>
  // CHECK-NEXT: |   |       `-DeclRefExpr {{.+}} 'unsigned int' lvalue Var {{.+}} 'uval' 'unsigned int'

  ptr_a = __arm_mte_get_tag(ptr_a);
  // CHECK-NEXT: |   |-BinaryOperator {{.+}} 'struct A *' '='
  // CHECK-NEXT: |   | |-DeclRefExpr {{.+}} 'struct A *' lvalue Var {{.+}} 'ptr_a' 'struct A *'
  // CHECK-NEXT: |   | `-CallExpr {{.+}} 'struct A *'
  // CHECK-NEXT: |   |   |-ImplicitCastExpr {{.+}} 'void *(*)(void *)' <BuiltinFnToFnPtr>
  // CHECK-NEXT: |   |   | `-DeclRefExpr {{.+}} '<builtin fn type>' Function {{.+}} '__builtin_arm_ldg' 'void *(void *)'
  // CHECK-NEXT: |   |   `-ImplicitCastExpr {{.+}} 'struct A *' <LValueToRValue>
  // CHECK-NEXT: |   |     `-DeclRefExpr {{.+}} 'struct A *' lvalue Var {{.+}} 'ptr_a' 'struct A *'

  __arm_mte_set_tag(ptr_a);
  // CHECK-NEXT: |   |-CallExpr {{.+}} 'void'
  // CHECK-NEXT: |   | |-ImplicitCastExpr {{.+}} 'void (*)(void *)' <BuiltinFnToFnPtr>
  // CHECK-NEXT: |   | | `-DeclRefExpr {{.+}} '<builtin fn type>' Function {{.+}} '__builtin_arm_stg' 'void (void *)'
  // CHECK-NEXT: |   | `-ImplicitCastExpr {{.+}} 'struct A *' <LValueToRValue>
  // CHECK-NEXT: |   |   `-DeclRefExpr {{.+}} 'struct A *' lvalue Var {{.+}} 'ptr_a' 'struct A *'

  pdval = __arm_mte_ptrdiff(ptr_a, ptr_a2);
  // CHECK-NEXT: |   `-BinaryOperator {{.+}} 'ptrdiff_t':'long' '='
  // CHECK-NEXT: |     |-DeclRefExpr {{.+}} 'ptrdiff_t':'long' lvalue Var {{.+}} 'pdval' 'ptrdiff_t':'long'
  // CHECK-NEXT: |     `-CallExpr {{.+}} 'long'
  // CHECK-NEXT: |       |-ImplicitCastExpr {{.+}} 'long (*)(void *, void *)' <BuiltinFnToFnPtr>
  // CHECK-NEXT: |       | `-DeclRefExpr {{.+}} '<builtin fn type>' Function {{.+}} '__builtin_arm_subp' 'long (void *, void *)'
  // CHECK-NEXT: |       |-ImplicitCastExpr {{.+}} 'struct A *' <LValueToRValue>
  // CHECK-NEXT: |       | `-DeclRefExpr {{.+}} 'struct A *' lvalue Var {{.+}} 'ptr_a' 'struct A *'
  // CHECK-NEXT: |       `-ImplicitCastExpr {{.+}} 'struct A *' <LValueToRValue>
  // CHECK-NEXT: |         `-DeclRefExpr {{.+}} 'struct A *' lvalue Var {{.+}} 'ptr_a2' 'struct A *'
}
