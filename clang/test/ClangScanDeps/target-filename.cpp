// RUN: rm -rf %t.dir
// RUN: rm -rf %t.cdb
// RUN: mkdir -p %t.dir
// RUN: cp %s %t.dir/target-filename_input.cpp
// RUN: mkdir %t.dir/Inputs
// RUN: cp %S/Inputs/header.h %t.dir/Inputs/header.h
// RUN: sed -e "s|DIR|%/t.dir|g" %S/Inputs/target-filename-cdb.json > %t.cdb
// RUN: clang-scan-deps -compilation-database %t.cdb -j 1 | FileCheck %s

// CHECK: target-filename_input.o:
// CHECK-NEXT: target-filename_input.cpp

// CHECK-NEXT: a.o:
// CHECK-NEXT: target-filename_input.cpp

// CHECK-NEXT: b.o:
// CHECK-NEXT: target-filename_input.cpp

// CHECK-NEXT: last.o:
// CHECK-NEXT: target-filename_input.cpp
