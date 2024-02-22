// RUN: rm -rf %t && mkdir -p %t
// RUN: llvm-cas --cas %t/cas --ingest %s > %t/casid
//
// RUN: %clang -cc1 -triple x86_64-apple-macos11 -fcas-backend \
// RUN:   -fcas-path %t/cas -fcas-fs @%t/casid -fcache-compile-job \
// RUN:   -Rcompile-job-cache %s -emit-obj -o %t/output.o \
// RUN:   -debug-info-kind=standalone -dwarf-version=4 -debugger-tuning=lldb \
// RUN:   -dependency-file %t/deps.d -MT %t/output.o 2>&1 \
// RUN:   | FileCheck %s --allow-empty --check-prefix=CACHE-MISS
//
// RUN: ls %t/output.o && rm %t/output.o
// RUN: ls %t/deps.d && mv %t/deps.d %t/deps.d.orig
//
// RUN: CLANG_CAS_BACKEND_SAVE_CASID_FILE=1 %clang -cc1 \
// RUN:   -triple x86_64-apple-macos11 -fcas-backend \
// RUN:   -fcas-path %t/cas -fcas-fs @%t/casid -fcache-compile-job \
// RUN:   -Rcompile-job-cache %s -emit-obj -o %t/output.o \
// RUN:   -debug-info-kind=standalone -dwarf-version=4 -debugger-tuning=lldb \
// RUN:   -dependency-file %t/deps.d -MT %t/output.o 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CACHE-HIT
//
// RUN: ls %t/output.o
// RUN: diff -u %t/deps.d %t/deps.d.orig
// RUN: llvm-cas-dump --cas %t/cas --casid-file \
// RUN:   --object-stats - %t/output.o.casid
//
// CACHE-HIT: remark: compile job cache hit
// CACHE-MISS-NOT: remark: compile job cache hit

void test(void) {}

int test1(void) {
  return 0;
}
