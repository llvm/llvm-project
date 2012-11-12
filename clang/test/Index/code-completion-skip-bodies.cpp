
// This is to make sure we skip function bodies.
void func_to_skip() {
  undeclared1 = 0;
}

struct S { int x; };

void func(S *s) {
  undeclared2 = 0;
  s->x = 0;
}

// RUN: c-index-test -code-completion-at=%s:11:6 %s 2>&1 | FileCheck %s
// CHECK-NOT: error: use of undeclared identifier 'undeclared1'
// CHECK: error: use of undeclared identifier 'undeclared2'
// CHECK: FieldDecl:{ResultType int}{TypedText x}

// FIXME: Investigating
// XFAIL: cygwin,mingw32,win32
