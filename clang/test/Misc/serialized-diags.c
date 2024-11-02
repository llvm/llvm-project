void foo(void) {
  int voodoo;
  voodoo = voodoo + 1;
}

void bar(void) {
  int dragon;
  dragon = dragon + 1
}

// Test handling of FixIts that only remove text.
int baz(void);
void qux(int x) {
  if ((x == baz()))
   return;
}

// Test handling of macros.
void taz(int x, int y);
#define false 0
void testMacro(void) {
  taz(0, 0, false);
}

// Test handling of issues from #includes.
#include "serialized-diags.h"

// Test handling of warnings that have empty fixits.
void rdar11040133(void) {
  unsigned x;
}

// RUN: rm -f %t
// RUN: not %clang -Wall -fsyntax-only %s --serialize-diagnostics %t.diag > /dev/null 2>&1
// RUN: c-index-test -read-diagnostics %t.diag > %t 2>&1
// RUN: FileCheck --input-file=%t %s

// This test case tests that we can handle multiple diagnostics which contain
// FIXITs at different levels (one at the note, another at the main diagnostic).

// CHECK: {{.*[/\\]}}serialized-diags.c:3:12: warning: variable 'voodoo' is uninitialized when used here [-Wuninitialized]
// CHECK: Range: {{.*[/\\]}}serialized-diags.c:3:12 {{.*[/\\]}}serialized-diags.c:3:18
// CHECK: +-{{.*[/\\]}}serialized-diags.c:2:13: note: initialize the variable 'voodoo' to silence this warning []
// CHECK: +-FIXIT: ({{.*[/\\]}}serialized-diags.c:2:13 - {{.*[/\\]}}serialized-diags.c:2:13): " = 0"
// CHECK: {{.*[/\\]}}serialized-diags.c:8:22: error: expected ';' after expression []
// CHECK: FIXIT: ({{.*[/\\]}}serialized-diags.c:8:22 - {{.*[/\\]}}serialized-diags.c:8:22): ";"
// CHECK: {{.*[/\\]}}serialized-diags.c:14:10: warning: equality comparison with extraneous parentheses [-Wparentheses-equality]
// CHECK: Range: {{.*[/\\]}}serialized-diags.c:14:8 {{.*[/\\]}}serialized-diags.c:14:18
// CHECK: +-{{.*[/\\]}}serialized-diags.c:14:10: note: remove extraneous parentheses around the comparison to silence this warning []
// CHECK: +-FIXIT: ({{.*[/\\]}}serialized-diags.c:14:7 - {{.*[/\\]}}serialized-diags.c:14:8): ""
// CHECK: +-FIXIT: ({{.*[/\\]}}serialized-diags.c:14:18 - {{.*[/\\]}}serialized-diags.c:14:19): ""
// CHECK: +-{{.*[/\\]}}serialized-diags.c:14:10: note: use '=' to turn this equality comparison into an assignment []
// CHECK: +-FIXIT: ({{.*[/\\]}}serialized-diags.c:14:10 - {{.*[/\\]}}serialized-diags.c:14:12): "="
// CHECK: {{.*[/\\]}}serialized-diags.c:22:13: error: too many arguments to function call, expected 2, have 3 []
// CHECK: Range: {{.*[/\\]}}serialized-diags.c:22:3 {{.*[/\\]}}serialized-diags.c:22:6
// CHECK: Range: {{.*[/\\]}}serialized-diags.c:22:13 {{.*[/\\]}}serialized-diags.c:22:18
// CHECK: +-{{.*[/\\]}}serialized-diags.c:20:15: note: expanded from macro 'false' []
// CHECK: +-Range: {{.*[/\\]}}serialized-diags.c:20:15 {{.*[/\\]}}serialized-diags.c:20:16
// CHECK: +-{{.*[/\\]}}serialized-diags.c:19:6: note: 'taz' declared here []
// CHECK: {{.*[/\\]}}serialized-diags.h:5:7: error: incompatible integer to pointer conversion initializing 'char *' with an expression of type 'int' [-Wint-conversion]
// CHECK: Range: {{.*[/\\]}}serialized-diags.h:5:16 {{.*[/\\]}}serialized-diags.h:5:17
// CHECK: +-{{.*[/\\]}}serialized-diags.c:26:10: note: in file included from {{.*[/\\]}}serialized-diags.c:26: []
// CHECK: Number FIXITs = 0
// CHECK: {{.*[/\\]}}serialized-diags.c:30:12: warning: unused variable 'x'
// CHECK: Number FIXITs = 0
// CHECK: Number of diagnostics: 6
