

// RUN: %clang_cc1 -fbounds-safety -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s | FileCheck %s

#include <ptrcheck.h>

int i;
int arr[2];
void Foo(int *__counted_by(len) ptr, unsigned len);

__auto_type glob_from_addrof = &i;            /* int *__bidi_indexable */
__auto_type glob_from_arrdecay = arr;         /* int *__bidi_indexable */
__auto_type glob_from_addr_of_arr = &arr[1];  /* int *__bidi_indexable */
__auto_type glob_from_fundecay = Foo;         /* int *__single */

void Test() {
  int *__single single_ptr;
  int *__indexable index_ptr;
  int *__unsafe_indexable unsafe_ptr;

  __auto_type local_from_arrdecay = arr;        /* int *__bidi_indexable */
  __auto_type local_from_addrof = &i;           /* int *__bidi_indexable */
  __auto_type local_from_addr_of_arr = &arr[0]; /* int *__bidi_indexable */
  __auto_type local_from_fundecay = Foo;        /* void(*__single)(int *__counted_by(), unsigned) */
  __auto_type local_from_single = single_ptr;   /* int *__single */
  __auto_type local_from_index = index_ptr;     /* int *__indexable */
  __auto_type local_from_unsafe = unsafe_ptr;   /* int *__unsafe_indexable */
}

// CHECK:     `-FunctionDecl {{.*}} <line:17:1, line:29:1> line:17:6 Test 'void ()'
// CHECK:         |-DeclStmt {{.*}} <line:22:3, col:40>
// CHECK-NEXT:    | `-VarDecl {{.*}} <col:3, col:37> col:15 local_from_arrdecay 'int *__bidi_indexable' cinit
// CHECK-NEXT:    |   `-ImplicitCastExpr {{.*}} <col:37> 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT:    |     `-DeclRefExpr {{.*}} <col:37> 'int[2]' lvalue Var {{.*}} 'arr' 'int[2]'
// CHECK-NEXT:    |-DeclStmt {{.*}} <line:23:3, col:37>
// CHECK-NEXT:    | `-VarDecl {{.*}} <col:3, col:36> col:15 local_from_addrof 'int *__bidi_indexable' cinit
// CHECK-NEXT:    |   `-UnaryOperator {{.*}} <col:35, col:36> 'int *__bidi_indexable' prefix '&' cannot overflow
// CHECK-NEXT:    |     `-DeclRefExpr {{.*}} <col:36> 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT:    |-DeclStmt {{.*}} <line:24:3, col:47>
// CHECK-NEXT:    | `-VarDecl {{.*}} <col:3, col:46> col:15 local_from_addr_of_arr 'int *__bidi_indexable' cinit
// CHECK-NEXT:    |   `-UnaryOperator {{.*}} <col:40, col:46> 'int *__bidi_indexable' prefix '&' cannot overflow
// CHECK-NEXT:    |     `-ArraySubscriptExpr {{.*}} <col:41, col:46> 'int' lvalue
// CHECK-NEXT:    |       |-ImplicitCastExpr {{.*}} <col:41> 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT:    |       | `-DeclRefExpr {{.*}} <col:41> 'int[2]' lvalue Var {{.*}} 'arr' 'int[2]'
// CHECK-NEXT:    |       `-IntegerLiteral {{.*}} <col:45> 'int' 0
// CHECK-NEXT:    |-DeclStmt {{.*}} <line:25:3, col:40>
// CHECK-NEXT:    | `-VarDecl {{.*}} <col:3, col:37> col:15 local_from_fundecay 'void (*__single)(int *__single __counted_by(len), unsigned int)' cinit
// CHECK-NEXT:    |   `-ImplicitCastExpr {{.*}} <col:37> 'void (*__single)(int *__single __counted_by(len), unsigned int)' <FunctionToPointerDecay>
// CHECK-NEXT:    |     `-DeclRefExpr {{.*}} <col:37> 'void (int *__single __counted_by(len), unsigned int)' Function {{.*}} 'Foo' 'void (int *__single __counted_by(len), unsigned int)'
// CHECK-NEXT:    |-DeclStmt {{.*}} <line:26:3, col:45>
// CHECK-NEXT:    | `-VarDecl {{.*}} <col:3, col:35> col:15 local_from_single 'int *__single' cinit
// CHECK-NEXT:    |   `-ImplicitCastExpr {{.*}} <col:35> 'int *__single' <LValueToRValue>
// CHECK-NEXT:    |     `-DeclRefExpr {{.*}} <col:35> 'int *__single' lvalue Var {{.*}} 'single_ptr' 'int *__single'
// CHECK-NEXT:    |-DeclStmt {{.*}} <line:27:3, col:43>
// CHECK-NEXT:    | `-VarDecl {{.*}} <col:3, col:34> col:15 local_from_index 'int *__indexable' cinit
// CHECK-NEXT:    |   `-ImplicitCastExpr {{.*}} <col:34> 'int *__indexable' <LValueToRValue>
// CHECK-NEXT:    |     `-DeclRefExpr {{.*}} <col:34> 'int *__indexable' lvalue Var {{.*}} 'index_ptr' 'int *__indexable'
// CHECK-NEXT:    `-DeclStmt {{.*}} <line:28:3, col:45>
// CHECK-NEXT:      `-VarDecl {{.*}} <col:3, col:35> col:15 local_from_unsafe 'int *__unsafe_indexable' cinit
// CHECK-NEXT:        `-ImplicitCastExpr {{.*}} <col:35> 'int *__unsafe_indexable' <LValueToRValue>
// CHECK-NEXT:          `-DeclRefExpr {{.*}} <col:35> 'int *__unsafe_indexable' lvalue Var {{.*}} 'unsafe_ptr' 'int *__unsafe_indexable'
