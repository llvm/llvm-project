// RUN: rm -rf %t.idx
// RUN: %clang_cc1 %s -index-store-path %t.idx -index-store-compress
// RUN: c-index-test core -print-unit %t.idx | FileCheck --check-prefix=UNIT %s
// RUN: c-index-test core -print-record %t.idx | FileCheck --check-prefix=RECORD %s

// UNIT: main-path: {{.*}}/compress-index-store.c
// RECORD: [[@LINE+1]]:6 | function/C | c:@F@foo | Decl | rel: 0
void foo(int *p);
