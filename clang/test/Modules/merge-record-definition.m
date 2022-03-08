// UNSUPPORTED: -zos, -aix
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: %clang_cc1 -emit-llvm -o %t/test.bc -F%S/Inputs/merge-record-definition %s \
// RUN:            -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache

// Test a case when a struct definition is present in two different modules.

#import <RecordDef/RecordDef.h>

void bibi(void) {
  Buffer buf;
  buf.b = 1;
  AnonymousStruct strct;
  strct.x = 1;
  UnionRecord rec;
  rec.u = 1;
}

#import <RecordDefCopy/RecordDefCopy.h>

void mbap(void) {
  Buffer buf;
  buf.c = 2;
  AnonymousStruct strct;
  strct.y = 2;
  UnionRecord rec;
  rec.v = 2;
}
