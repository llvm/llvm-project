// By default, both the error and the note show a snippet.
// RUN: not %clang_cc1 -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=SNIPPET
// SNIPPET: error: redefinition of 'x'
// SNIPPET-NEXT: {{[0-9]+}} | float x;
// SNIPPET-NEXT: | {{.*}}^
// SNIPPET-NEXT: note: previous definition is here
// SNIPPET-NEXT: {{[0-9]+}} | int x;
// SNIPPET-NEXT: | {{.*}}^

// With -fno-diagnostics-show-note-snippets, the error keeps its snippet but the
// note only shows its message line.
// RUN: not %clang_cc1 -fsyntax-only -fno-diagnostics-show-note-snippets %s 2>&1 | FileCheck %s --check-prefix=NOSNIPPET
// NOSNIPPET: error: redefinition of 'x'
// NOSNIPPET-NEXT: {{[0-9]+}} | float x;
// NOSNIPPET-NEXT: | {{.*}}^
// NOSNIPPET-NEXT: note: previous definition is here
// NOSNIPPET-NOT: int x;
// NOSNIPPET: error: no matching function for call to 'f'
// NOSNIPPET-NEXT: {{[0-9]+}} | f();
// NOSNIPPET-NEXT: | ^
// NOSNIPPET-NEXT: note: candidate function not viable
// NOSNIPPET-NOT: void f(int);


// With -fno-diagnostics-show-note-snippets, parseable fixits on notes are still
// emitted.
// RUN: not %clang_cc1 -fsyntax-only -fno-diagnostics-show-note-snippets -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s --check-prefix=FIXIT
// FIXIT: warning: using the result of an assignment
// FIXIT: note: place parentheses around the assignment
// FIXIT-NEXT: fix-it:{{.*}}diag-note-snippets.cpp

int x;
float x;

void f(int);

void g(int a) {
  f();
  if (a = 5) {
  }
}
