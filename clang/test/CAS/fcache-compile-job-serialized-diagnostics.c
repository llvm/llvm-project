// RUN: rm -rf %t && mkdir -p %t
// RUN: llvm-cas --cas %t/cas --ingest --data %s > %t/casid

// RUN: %clang -cc1 -triple x86_64-apple-macos11 -fcas-path %t/cas -faction-cache-path %t/cache \
// RUN:   -fcas-fs @%t/casid -fcache-compile-job \
// RUN:   -Wimplicit-function-declaration \
// RUN:   -Wno-error=implicit-function-declaration \
// RUN:   -Rcompile-job-cache -emit-obj -o %t/output.o \
// RUN:   -serialize-diagnostic-file %t/diags %s 2>&1 \
// RUN:   | FileCheck %s --allow-empty --check-prefix=CACHE-MISS

// RUN: c-index-test -read-diagnostics %t/diags 2>&1 | FileCheck %s --check-prefix=SERIALIZED-MISS --check-prefix=SERIALIZED-COMMON

// RUN: ls %t/output.o && rm %t/output.o

// RUN: %clang -cc1 -triple x86_64-apple-macos11 -fcas-path %t/cas -faction-cache-path %t/cache \
// RUN:   -fcas-fs @%t/casid -fcache-compile-job \
// RUN:   -Wimplicit-function-declaration \
// RUN:   -Wno-error=implicit-function-declaration \
// RUN:   -Rcompile-job-cache -emit-obj -o %t/output.o \
// RUN:   -serialize-diagnostic-file %t/diags %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CACHE-HIT

// RUN: c-index-test -read-diagnostics %t/diags 2>&1 | FileCheck %s --check-prefix=SERIALIZED-HIT --check-prefix=SERIALIZED-COMMON

// CACHE-HIT: remark: compile job cache hit
// CACHE-HIT: warning: some warning

// CACHE-MISS: remark: compile job cache miss
// CACHE-MISS: warning: some warning

// SERIALIZED-HIT: warning: compile job cache hit
// SERIALIZED-MISS: warning: compile job cache miss
// SERIALIZED-COMMON: warning: some warning
// SERIALIZED-COMMON: Number of diagnostics: 2

// Make sure warnings are merged with driver ones.
// Using normal compilation as baseline.
// RUN: %clang -target x86_64-apple-macos11 -c %s -o %t/t.o -Wl,-none --serialize-diagnostics %t/t1.diag \
// RUN:   2>&1 | FileCheck %s -check-prefix=WARN
// RUN: env LLVM_CACHE_CAS_PATH=%t/cas CLANG_CACHE_USE_CASFS_DEPSCAN=1 %clang-cache \
// RUN:   %clang -target x86_64-apple-macos11 -c %s -o %t/t.o -Wl,-none --serialize-diagnostics %t/t2.diag \
// RUN:   2>&1 | FileCheck %s -check-prefix=WARN
// RUN: env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache \
// RUN:   %clang -target x86_64-apple-macos11 -c %s -o %t/t.o -Wl,-none --serialize-diagnostics %t/t2.inc.diag \
// RUN:   2>&1 | FileCheck %s -check-prefix=WARN
// RUN: diff %t/t1.diag %t/t2.diag
// RUN: diff %t/t1.diag %t/t2.inc.diag

// Try again with cache hit.
// RUN: rm %t/t2.diag
// RUN: env LLVM_CACHE_CAS_PATH=%t/cas CLANG_CACHE_USE_CASFS_DEPSCAN=1 %clang-cache \
// RUN:   %clang -target x86_64-apple-macos11 -c %s -o %t/t.o -Wl,-none --serialize-diagnostics %t/t2.diag
// RUN: env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache \
// RUN:   %clang -target x86_64-apple-macos11 -c %s -o %t/t.o -Wl,-none --serialize-diagnostics %t/t2.inc.diag \
// RUN:   2>&1 | FileCheck %s -check-prefix=WARN
// RUN: diff %t/t1.diag %t/t2.diag
// RUN: diff %t/t1.diag %t/t2.inc.diag

// WARN: warning: -Wl,-none: 'linker' input unused
// WARN: warning: some warning

#warning some warning
