// RUN: rm -rf %t && mkdir -p %t
// RUN: llvm-cas --cas %t/cas --ingest --data %s > %t/casid
//
// RUN: %clang -cc1 -triple x86_64-apple-macos11 \
// RUN:   -fcas-path %t/cas -fcas-fs @%t/casid -fcache-compile-job \
// RUN:   -Rcompile-job-cache -emit-obj -o %t/output.o \
// RUN:   -dependency-file %t/deps.d -MT %t/output.o 2>&1 \
// RUN:   | FileCheck %s --allow-empty --check-prefix=CACHE-MISS
//
// RUN: ls %t/output.o && rm %t/output.o
// RUN: ls %t/deps.d && mv %t/deps.d %t/deps.d.orig
//
// RUN: %clang -cc1 -triple x86_64-apple-macos11 \
// RUN:   -fcas-path %t/cas -fcas-fs @%t/casid -fcache-compile-job \
// RUN:   -Rcompile-job-cache -emit-obj -o %t/output.o \
// RUN:   -dependency-file %t/deps.d -MT %t/output.o 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CACHE-HIT
//
// RUN: ls %t/output.o
// RUN: diff -u %t/deps.d %t/deps.d.orig
//
// CACHE-HIT: remark: compile job cache hit
// CACHE-MISS-NOT: remark: compile job cache hit
