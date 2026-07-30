
// This is to make sure we skip function bodies.
void func_to_skip() {
  undeclared1 = 0;
}

struct S { int x; };

void func(S *s) {
  undeclared2 = 0;
  s->x = 0;
}

// RUN: c-index-test -code-completion-at=%s:11:6 %s 2> %t.stderr | FileCheck %s --check-prefix=STDOUT
// RUN: FileCheck --input-file=%t.stderr --check-prefix=STDERR %s

// STDOUT: FieldDecl:{ResultType int}{TypedText x}

// STDERR-NOT: error: use of undeclared identifier 'undeclared1'
// STDERR:     error: use of undeclared identifier 'undeclared2'
