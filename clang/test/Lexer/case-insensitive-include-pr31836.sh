// REQUIRES: case-insensitive-filesystem

// RUN: mkdir -p %t
// RUN: touch %t/case-insensitive-include-pr31836.h
// RUN: echo "#include \"\\\\\\\\?\\\\%t/Case-Insensitive-Include-Pr31836.h\"" | not %clang_cc1 -E - 2>&1 | FileCheck %s

// CHECK: error: {{.*}}file not found, did you mean
// CHECK: warning: non-portable path to file
// CHECK-SAME: /case-insensitive-include-pr31836.h
