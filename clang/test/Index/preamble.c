#include "prefix.h"
#include "preamble.h"
#include "preamble-with-error.h"

#define MACRO_UNUSED 1
#define MACRO_USED 2

int wibble(int);

void f(int x) {
  x = MACRO_USED
}
// RUN: c-index-test -write-pch %t.pch -x c-header %S/Inputs/prefix.h
// RUN: env CINDEXTEST_EDITING=1 c-index-test -test-load-source-reparse 5 local -I %S/Inputs -include %t %s -Wunused-macros 2> %t.stderr.txt | FileCheck %s
// RUN: FileCheck -check-prefix CHECK-DIAG %s < %t.stderr.txt
// CHECK: preamble.h:1:12: FunctionDecl=bar:1:12 (Definition) Extent=[1:1 - 6:2]
// CHECK: preamble.h:4:3: BinaryOperator= Extent=[4:3 - 4:13]
// CHECK: preamble.h:4:3: DeclRefExpr=ptr:2:8 Extent=[4:3 - 4:6]
// CHECK: preamble.h:4:9: UnexposedExpr=ptr1:3:10 Extent=[4:9 - 4:13]
// CHECK: preamble.h:4:9: DeclRefExpr=ptr1:3:10 Extent=[4:9 - 4:13]
// CHECK: preamble.h:5:10: IntegerLiteral= Extent=[5:10 - 5:11]
// CHECK: preamble.c:8:5: FunctionDecl=wibble:8:5 Extent=[8:1 - 8:16]
// CHECK: preamble.c:8:15: ParmDecl=:8:15 (Definition) Extent=[8:12 - 8:15]
// CHECK-DIAG: preamble.h:4:7:{4:9-4:13}: warning: incompatible pointer types assigning to 'int *' from 'float *'
// FIXME: Should see:
//     preamble.c:5:9: warning: macro is not used
// CHECK-DIAG-NOT: preamble.c:6:9: warning: macro is not used
// RUN: env CINDEXTEST_EDITING=1 c-index-test -code-completion-at=%s:11:1 -I %S/Inputs -include %t %s 2> %t.stderr.txt | FileCheck -check-prefix CHECK-CC %s
// CHECK-CC: FunctionDecl:{ResultType int}{TypedText bar}{LeftParen (}{Placeholder int i}{RightParen )} (50)
// CHECK-CC: FunctionDecl:{ResultType void}{TypedText f}{LeftParen (}{Placeholder int x}{RightParen )} (50)
// CHECK-CC: FunctionDecl:{ResultType int}{TypedText foo}{LeftParen (}{Placeholder int}{RightParen )} (50)
// CHECK-CC: FunctionDecl:{ResultType int}{TypedText wibble}{LeftParen (}{Placeholder int}{RightParen )} (50)
