// RUN: %{to-crlf} %S/Inputs/comment_save_crlf.h > %t.h

// Verify that prepared header contains \r
// RUN: tr -d '\n' < %t.h | %{reveal-cr} | FileCheck %s --check-prefix=SANITY
// SANITY: <CR>

// RUN: %clang_cc1 -E -C -o %t.i %t.h
// RUN: %{reveal-cr} %t.i | FileCheck %s
// CHECK-NOT: <CR>
