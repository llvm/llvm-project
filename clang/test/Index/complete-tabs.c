// Test code-completion in the presence of tabs
struct Point { int x, y; };

void f(struct Point *p) {
	p->

// RUN: c-index-test -code-completion-at=%s:5:5 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: {TypedText x}
// CHECK-CC1: {TypedText y}
