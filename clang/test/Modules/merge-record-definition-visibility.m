// UNSUPPORTED: -zos, target={{.*}}-aix{{.*}}
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: %clang_cc1 -emit-llvm -o %t/test.bc -F%S/Inputs/merge-record-definition %s \
// RUN:            -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache

// Test a case when a struct definition is first imported as invisible and then as visible.

#import <RecordDefHidden/Visible.h>
#import <RecordDef/RecordDef.h>

void bibi(void) {
  Buffer buf;
  buf.b = 1;
  AnonymousStruct strct;
  strct.y = 1;
  UnionRecord rec;
  rec.u = 1;
}
