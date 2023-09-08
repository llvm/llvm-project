// REQUIRES: ondisk_cas

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: llvm-cas --cas %t.cas --ingest %t > %t/casid
//
// RUN: not %clang_cc1 -triple x86_64-apple-macos11 \
// RUN:   -fmodules -fmodule-name=A -fno-implicit-modules \
// RUN:   -emit-module %t/module.modulemap -o %t/A.pcm \
// RUN:   -fcas-path %t.cas -fcas-fs @%t/casid \
// RUN:   -fcache-compile-job -Rcompile-job-cache &> %t/A.out.txt
// RUN: FileCheck %s --input-file=%t/A.out.txt

// CHECK: remark: compile job cache miss
// CHECK: error: encountered non-reproducible token, caching failed
// CHECK: error: encountered non-reproducible token, caching failed
// CHECK: error: encountered non-reproducible token, caching failed

//--- module.modulemap
module A { header "A.h" }

//--- A.h
void getit(const char **p1, const char **p2, const char **p3) {
  *p1 = __DATE__;
  *p2 = __TIMESTAMP__;
  *p3 = __TIME__;
}
