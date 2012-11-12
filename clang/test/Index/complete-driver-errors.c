int *blah = 1;

int

// CHECK-RESULTS: NotImplemented:{TypedText const} (40)
// CHECK-RESULTS: NotImplemented:{TypedText restrict} (40)
// CHECK-RESULTS: NotImplemented:{TypedText volatile} (40)
// CHECK-DIAGS: error: invalid value '' in '-std='
// CHECK-DIAGS: complete-driver-errors.c:1:6:{1:13-1:14}: warning: incompatible integer to pointer conversion initializing 'int *' with an expression of type 'int'

// Test driver errors with code completion
// RUN: c-index-test -code-completion-at=%s:4:1 -std= %s 2> %t | FileCheck -check-prefix=CHECK-RESULTS %s
// RUN: FileCheck -check-prefix=CHECK-DIAGS %s < %t

// Test driver errors with parsing
// RUN: c-index-test -test-load-source all -std= %s 2> %t | FileCheck -check-prefix=CHECK-LOAD %s
// RUN: FileCheck -check-prefix=CHECK-DIAGS %s < %t
// CHECK-LOAD: complete-driver-errors.c:1:6: VarDecl=blah:1:6

// Test driver errors with code completion and precompiled preamble
// RUN: env CINDEXTEST_EDITING=1 c-index-test -code-completion-at=%s:4:1 -std= %s 2> %t | FileCheck -check-prefix=CHECK-RESULTS %s
// RUN: FileCheck -check-prefix=CHECK-DIAGS %s < %t
// RUN: env CINDEXTEST_EDITING=1 c-index-test -test-load-source all -std= %s 2> %t | FileCheck -check-prefix=CHECK-LOAD %s
// RUN: FileCheck -check-prefix=CHECK-DIAGS %s < %t
