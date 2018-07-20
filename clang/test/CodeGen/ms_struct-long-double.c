// RUN: %clang_cc1 -emit-llvm-only -triple i686-windows-gnu -fdump-record-layouts %s | FileCheck %s
// RUN: %clang_cc1 -emit-llvm-only -triple i686-linux -fdump-record-layouts -Wno-incompatible-ms-struct %s | FileCheck %s
// RUN: not %clang_cc1 -emit-llvm-only -triple i686-linux -fdump-record-layouts %s 2>&1 | FileCheck %s -check-prefix=ERROR

struct ldb_struct {
  char c;
  long double ldb;
} __attribute__((__ms_struct__));

struct ldb_struct a;

// CHECK:             0 | struct ldb_struct
// CHECK-NEXT:        0 |   char c
// CHECK-NEXT:        4 |   long double ldb
// CHECK-NEXT:          | [sizeof=16, align=4]

// ERROR: error: ms_struct may not produce Microsoft-compatible layouts with fundamental data types with sizes that aren't a power of two
