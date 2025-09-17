// Regression test for #139375
// Clang uses 1-based indexing for source locations given from the command-line.
// Verify that Clang rejects 0 as an invalid value for line or column number.

// RUN: not %clang_cc1 -fsyntax-only -code-completion-at=%s:0:1 %s -o - 2>&1 \
// RUN:     | FileCheck -check-prefix=CHECK-DIAG %s
// RUN: not %clang_cc1 -fsyntax-only -code-completion-at=%s:1:0 %s -o - 2>&1 \
// RUN:     | FileCheck -check-prefix=CHECK-DIAG %s

// CHECK-DIAG: error: invalid value '{{.*}}' in '-code-completion-at={{.*}}'
// CHECK-NEXT: hint: -code-completion-at=<file>:<line>:<column> requires <line> and <column> to be integers greater than zero
