void foo(int i);

// RUN: rm -rf %t/idx
// RUN: %clang_cc1 -index-store-path %t/idx %s -o %t.o
// RUN: touch %t.empty

// RUN: cp %t.empty $(find %t/idx -name "empty-unit.c*o*")
// RUN: not c-index-test core -print-unit %t/idx 2> %t.err
// RUN: FileCheck %s -input-file %t.err -check-prefix ERR-UNIT
// ERR-UNIT: error loading unit: empty file

// Also check for empty record files.
// RUN: rm -rf %t/idx2
// RUN: %clang_cc1 -index-store-path %t/idx2 %s -o %t.o
// RUN: cp %t.empty $(find %t/idx2 -name "empty-unit.c-*")
// RUN: not c-index-test core -print-record %t/idx2 2> %t2.err
// RUN: FileCheck %s -input-file %t2.err -check-prefix ERR-RECORD
// ERR-RECORD: error loading record: empty file

// REQUIRES: shell
