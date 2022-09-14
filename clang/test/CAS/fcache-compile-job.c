// RUN: rm -rf %t && mkdir -p %t
// RUN: llvm-cas --cas %t/cas --ingest --data %s > %t/casid
//
// RUN: %clang -cc1 -triple x86_64-apple-macos11 \
// RUN:   -fcas-path %t/cas -faction-cache-path %t/cache -fcas-fs @%t/casid -fcache-compile-job \
// RUN:   -Rcompile-job-cache -emit-obj %s -o %t/output.o 2>&1 \
// RUN:   | FileCheck %s --allow-empty --check-prefix=CACHE-MISS
// RUN: ls %t/output.o && rm %t/output.o
// RUN: %clang -cc1 -triple x86_64-apple-macos11 \
// RUN:   -fcas-path %t/cas -faction-cache-path %t/cache -fcas-fs @%t/casid -fcache-compile-job \
// RUN:   -Rcompile-job-cache -emit-obj %s -o %t/output.o 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CACHE-HIT
// RUN: ls %t/output.o && rm %t/output.o
// RUN: cd %t
// RUN: %clang -cc1 -triple x86_64-apple-macos11 \
// RUN:   -fcas-path %t/cas -faction-cache-path %t/cache -fcas-fs @%t/casid -fcache-compile-job \
// RUN:   -Rcompile-job-cache -emit-obj %s -o output.o 2>&1 \
// RUN:   | FileCheck %s --allow-empty --check-prefix=CACHE-HIT
// RUN: ls %t/output.o
//
// Check for a cache hit if the CAS moves:
// RUN: mv %t/cas %t/cas.moved
// RUN: %clang -cc1 -triple x86_64-apple-macos11 \
// RUN:   -fcas-path %t/cas.moved -faction-cache-path %t/cache -fcas-fs @%t/casid -fcache-compile-job \
// RUN:   -Rcompile-job-cache -emit-obj %s -o output.o 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CACHE-HIT
// RUN: ls %t/output.o
//
// Check for a handling error if the CAS is removed but not action cache.
// First need to ignest the input file so the compile cache can be constructed.
// RUN: llvm-cas --ingest --cas %t/cas.new --data %s
// RUN: not %clang -cc1 -triple x86_64-apple-macos11 \
// RUN:   -fcas-path %t/cas.new -faction-cache-path %t/cache -fcas-fs @%t/casid -fcache-compile-job \
// RUN:   -Rcompile-job-cache -emit-obj %s -o output.o 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CACHE-ERROR
// RUN: ls %t/output.o
//
// CACHE-HIT: remark: compile job cache hit
// CACHE-MISS-NOT: remark: compile job cache hit
// CACHE-ERROR: fatal error: caching backend error:
