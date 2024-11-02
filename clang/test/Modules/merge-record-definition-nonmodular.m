// UNSUPPORTED: -zos, target={{.*}}-aix{{.*}}
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: %clang_cc1 -emit-llvm -o %t/test.bc -F%S/Inputs/merge-record-definition %s \
// RUN:            -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache -fmodule-name=RecordDef
// RUN: %clang_cc1 -emit-llvm -o %t/test.bc -F%S/Inputs/merge-record-definition %s -DMODULAR_BEFORE_TEXTUAL \
// RUN:            -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache -fmodule-name=RecordDef

// Test a case when a struct definition once is included from a textual header and once from a module.

#ifdef MODULAR_BEFORE_TEXTUAL
  #import <RecordDefIncluder/RecordDefIncluder.h>
#else
  #import <RecordDef/RecordDef.h>
#endif

void bibi(void) {
  Buffer buf;
  buf.b = 1;
  AnonymousStruct strct;
  strct.x = 1;
  UnionRecord rec;
  rec.u = 1;
}

#ifdef MODULAR_BEFORE_TEXTUAL
  #import <RecordDef/RecordDef.h>
#else
  #import <RecordDefIncluder/RecordDefIncluder.h>
#endif

void mbap(void) {
  Buffer buf;
  buf.c = 2;
  AnonymousStruct strct;
  strct.y = 2;
  UnionRecord rec;
  rec.v = 2;
}
