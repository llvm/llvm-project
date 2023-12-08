
// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_CACHING=1 c-index-test \
// RUN:	    -code-completion-at=%S/Inputs/complete-at-EOF.c:4:1 %S/Inputs/complete-at-EOF.c | FileCheck -check-prefix=CHECK-EOF %s
// CHECK-EOF: macro definition:{TypedText CAKE}
// CHECK-EOF: TypedefDecl:{TypedText foo}

// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_CACHING=1 c-index-test \
// RUN:     -code-completion-at=%S/Inputs/complete-at-EOF.c:2:1 %S/Inputs/complete-at-EOF.c | FileCheck -check-prefix=CHECK-AFTER-PREAMBLE %s
// CHECK-AFTER-PREAMBLE: macro definition:{TypedText CAKE}
