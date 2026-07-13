// REQUIRES: system-windows

// RUN: %{to-crlf} %S/Inputs/comment_save_crlf.h > %t.h
// RUN: %clang_cc1 -E -C -o %t.i %t.h
// RUN: %{reveal-cr} %t.i | FileCheck %s
// CHECK-NOT: <CR>
