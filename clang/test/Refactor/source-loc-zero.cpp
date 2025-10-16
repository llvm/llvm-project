// Regression test for #139375
// Clang uses 1-based indexing for source locations given from the command-line.
// Verify that `clang-refactor` rejects 0 as an invalid value for line or column number.

// For range start:
// RUN: not clang-refactor local-rename -selection=%s:0:1-1:1 -new-name=test %s 2>&1 \
// RUN:     | FileCheck -check-prefix=CHECK-DIAG %s
// RUN: not clang-refactor local-rename -selection=%s:1:0-1:1 -new-name=test %s 2>&1 \
// RUN:     | FileCheck -check-prefix=CHECK-DIAG %s

// For range end:
// RUN: not clang-refactor local-rename -selection=%s:1:1-0:1 -new-name=test %s 2>&1 \
// RUN:     | FileCheck -check-prefix=CHECK-DIAG %s
// RUN: not clang-refactor local-rename -selection=%s:1:1-1:0 -new-name=test %s 2>&1 \
// RUN:     | FileCheck -check-prefix=CHECK-DIAG %s

// CHECK-DIAG: error: '-selection' option must be specified using <file>:<line>:<column> or <file>:<line>:<column>-<line>:<column> format, where <line> and <column> are integers greater than zero.
