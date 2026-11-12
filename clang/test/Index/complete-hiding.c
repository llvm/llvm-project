// Note: the run lines follow their respective tests, since line/column
// matter in this test.

struct StructA { };
struct StructB { };
struct StructC { };
int ValueA;
int ValueB;

void f() {

  int ValueA = 0;
  int StructA = 0;
  struct StructB { };
  
  struct StructA sa = { };
}

// RUN: c-index-test -code-completion-at=%s:16:3 %s > %t
// RUN: FileCheck -check-prefix=CHECK-CC1 -input-file=%t %s
// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_CACHING=1 c-index-test -code-completion-at=%s:16:3 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: VarDecl:{ResultType int}{TypedText StructA} (34)
// CHECK-CC1: VarDecl:{ResultType int}{TypedText ValueA} (34)
// CHECK-CC1-NOT: VarDecl:{ResultType int}{TypedText ValueA} (50)
// CHECK-CC1: VarDecl:{ResultType int}{TypedText ValueB} (50)
// RUN: c-index-test -code-completion-at=%s:16:10 %s > %t
// RUN: FileCheck -check-prefix=CHECK-CC2 -input-file=%t %s
// CHECK-CC2: StructDecl:{TypedText StructA} (50)
// CHECK-CC2-NOT: StructDecl:{TypedText StructB} (50)
// CHECK-CC2: StructDecl:{TypedText StructC} (50)
// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_CACHING=1 c-index-test -code-completion-at=%s:16:10 %s > %t
// RUN: FileCheck -check-prefix=CHECK-CC2 -input-file=%t %s
