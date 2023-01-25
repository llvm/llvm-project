/* RUN: %clang_cc1 -std=c89 -Wno-gcc-compat -ast-dump -o -  %s | FileCheck %s
   RUN: %clang_cc1 -std=c99 -ast-dump -o -  %s | FileCheck %s
   RUN: %clang_cc1 -std=c11 -ast-dump -o -  %s | FileCheck %s
   RUN: %clang_cc1 -std=c17 -ast-dump -o -  %s | FileCheck %s
   RUN: %clang_cc1 -std=c2x -ast-dump -o -  %s | FileCheck %s
 */

/* WG14 DR466: yes
 * Scope of a for loop control declaration
 */
int dr466(void) {
  for (int i = 0; ; ) {
    long i = 1;   /* valid C, invalid C++ */
    // ...
    return i;     /* (perhaps unexpectedly) returns 1 in C */
  }
}

/*
CHECK: FunctionDecl 0x{{.+}} dr466 'int (void)'
CHECK-NEXT: CompoundStmt
CHECK-NEXT: ForStmt
CHECK-NEXT: DeclStmt
CHECK-NEXT: VarDecl 0x{{.+}} {{.+}} i 'int'
CHECK: CompoundStmt
CHECK-NEXT: DeclStmt
CHECK-NEXT: VarDecl [[ACTUAL:0x.+]] <col:{{.+}}> col:{{.+}} used i 'long'
CHECK: ReturnStmt
CHECK: DeclRefExpr 0x{{.+}} <col:{{.+}}> 'long' lvalue Var [[ACTUAL]] 'i' 'long'
*/
