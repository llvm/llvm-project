// RUN: %clang_cc1 -remap-file "%s;%S/Inputs/remapped-file" -fsyntax-only %s 2>&1 | FileCheck -check-prefix=CHECK-EXIST %s
// RUN: %clang_cc1 -remap-file "%S/nonexistent.c;%S/Inputs/remapped-file" -fsyntax-only %S/nonexistent.c 2>&1 | FileCheck -check-prefix=CHECK-NONEXIST %s
// RUN: %clang_cc1 -remap-file "%S/nonexistent.c;%S/Inputs/remapped-file-2" -remap-file "%S/nonexistent.h;%S/Inputs/remapped-file-3" -fsyntax-only %S/nonexistent.c 2>&1 | FileCheck -check-prefix=CHECK-HEADER %s

// CHECK-EXIST: remap-file.c:1:28: warning: incompatible pointer types
// CHECK-NONEXIST: nonexistent.c:1:28: warning: incompatible pointer types
// CHECK-HEADER: nonexistent.c:3:23: warning: incompatible pointer types
int
