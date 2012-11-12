#include "prefix.h"
#include "preamble.h"
int wibble(int);

void f(int x) {
  
}
// RUN: c-index-test -write-pch %t.pch -x c-header %S/Inputs/prefix.h
// RUN: env CINDEXTEST_EDITING=1 c-index-test -test-load-source-reparse 5 local -I %S/Inputs -include %t %s 2> %t.stderr.txt | FileCheck %s
// RUN: FileCheck -check-prefix CHECK-DIAG %s < %t.stderr.txt
// CHECK: preamble.h:1:12: FunctionDecl=bar:1:12 (Definition) Extent=[1:1 - 6:2]
// CHECK: preamble.h:4:3: BinaryOperator= Extent=[4:3 - 4:13]
// CHECK: preamble.h:4:3: DeclRefExpr=ptr:2:8 Extent=[4:3 - 4:6]
// CHECK: preamble.h:4:9: UnexposedExpr=ptr1:3:10 Extent=[4:9 - 4:13]
// CHECK: preamble.h:4:9: DeclRefExpr=ptr1:3:10 Extent=[4:9 - 4:13]
// CHECK: preamble.h:5:10: IntegerLiteral= Extent=[5:10 - 5:11]
// CHECK: preamble.c:3:5: FunctionDecl=wibble:3:5 Extent=[3:1 - 3:16]
// CHECK: preamble.c:3:15: ParmDecl=:3:15 (Definition) Extent=[3:12 - 3:16]
// CHECK-DIAG: preamble.h:4:7:{4:9-4:13}: warning: incompatible pointer types assigning to 'int *' from 'float *'
// RUN: env CINDEXTEST_EDITING=1 c-index-test -code-completion-at=%s:6:1 -I %S/Inputs -include %t %s 2> %t.stderr.txt | FileCheck -check-prefix CHECK-CC %s
// CHECK-CC: FunctionDecl:{ResultType int}{TypedText bar}{LeftParen (}{Placeholder int i}{RightParen )} (50)
// CHECK-CC: FunctionDecl:{ResultType void}{TypedText f}{LeftParen (}{Placeholder int x}{RightParen )} (50)
// CHECK-CC: FunctionDecl:{ResultType int}{TypedText foo}{LeftParen (}{Placeholder int}{RightParen )} (50)
// CHECK-CC: FunctionDecl:{ResultType int}{TypedText wibble}{LeftParen (}{Placeholder int}{RightParen )} (50)
