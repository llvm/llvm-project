/* RUN: %clang_cc1 -std=c89 -ast-dump -o -  %s | FileCheck %s
   RUN: %clang_cc1 -std=c99 -ast-dump -o -  %s | FileCheck %s
   RUN: %clang_cc1 -std=c11 -ast-dump -o -  %s | FileCheck %s
   RUN: %clang_cc1 -std=c17 -ast-dump -o -  %s | FileCheck %s
   RUN: %clang_cc1 -std=c2x -ast-dump -o -  %s | FileCheck %s
 */

/* WG14 DR253: yes
 * "overriding" in designated initializers
 */
struct fred {
  char s [6];
  int n;
};

struct fred y [] = { { { "abc" }, 1 }, [0] = { .s[0] = 'q' } };

/* Ensure that y[0] is initialized as if by the initializer { 'q' }. */

// CHECK: VarDecl 0x{{.*}} <line:16:1, col:62> col:13 y 'struct fred[1]' cinit
// CHECK-NEXT: InitListExpr 0x{{.*}} <col:20, col:62> 'struct fred[1]'
// CHECK-NEXT: InitListExpr 0x{{.*}} <col:46, col:60> 'struct fred':'struct fred'
// CHECK-NEXT: InitListExpr 0x{{.*}} <col:50, col:56> 'char[6]'
// CHECK-NEXT: array_filler
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: CharacterLiteral 0x{{.*}} <col:56> 'int' 113
// CHECK-NEXT: ImplicitValueInitExpr 0x{{.*}} <<invalid sloc>> 'int'
