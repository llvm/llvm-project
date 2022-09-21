// Check that fatal errors from cache-related output paths show up.

// REQUIRES: shell

// RUN: rm -rf %t && mkdir -p %t/a
// RUN: llvm-cas --cas %t/cas --ingest --data %s > %t/casid

// RUN: %clang -cc1 -triple x86_64-apple-macos11 \
// RUN:   -fcas-path %t/cas -faction-cache-path %t/cache -fcas-fs @%t/casid -fcache-compile-job \
// RUN:   -Rcompile-job-cache -emit-obj %s -o %t/a/output.o 2>&1 \
// RUN:   | FileCheck %s --allow-empty --check-prefix=CACHE-MISS

// RUN: mkdir %t/b
// RUN: chmod -w %t/b

// RUN: not %clang -cc1 -triple x86_64-apple-macos11 \
// RUN:   -fcas-path %t/cas -faction-cache-path %t/cache -fcas-fs @%t/casid -fcache-compile-job \
// RUN:   -Rcompile-job-cache -emit-obj %s -o %t/b/output.o 2>&1 \
// RUN:   | FileCheck %s --allow-empty --check-prefixes=CACHE-HIT,ERROR

// CACHE-MISS: remark: compile job cache miss
// CACHE-HIT: remark: compile job cache hit
// ERROR: fatal error: error in backend: Permission denied
