// REQUIRES: case-insensitive-filesystem
// UNSUPPORTED: system-windows

// RUN: mkdir -p %T
// RUN: touch %T/case-insensitive-include-pr31836.h
// RUN: echo "#include \"%T/Case-Insensitive-Include-Pr31836.h\"" | %clang_cc1 -E - 2>&1 | FileCheck %s

// CHECK: warning: non-portable path to file
// CHECK-SAME: /case-insensitive-include-pr31836.h
