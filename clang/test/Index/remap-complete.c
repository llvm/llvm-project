// RUN: c-index-test -code-completion-at=%s:6:2 -remap-file="%s;%S/Inputs/remap-complete-to.c" %s | FileCheck %s

// CHECK: FunctionDecl:{ResultType int}{TypedText f0}{LeftParen (}
void f() { }
