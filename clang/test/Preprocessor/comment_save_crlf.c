// REQUIRES: system-windows

// RUN: %clang_cc1 -E -C -o %t %S/Inputs/comment_save_crlf.h
// RUN: FileCheck --strict-whitespace --match-full-lines --input-file=%t %s

// CHECK:int a;
// CHECK-NEXT:/* block comment
// CHECK-NEXT:   spanning multiple
// CHECK-NEXT:   CRLF-terminated lines */
// CHECK-NEXT:int b;
