// RUN: rm -rf %t && mkdir -p %t
// RUN: llvm-cas --cas %t/cas --ingest --data %s > %t/casid

// RUN: %clang -cc1 -triple x86_64-apple-macos11 -fcas-path %t/cas \
// RUN:   -fcas-fs @%t/casid -fcache-compile-job \
// RUN:   -Wimplicit-function-declaration \
// RUN:   -Wno-error=implicit-function-declaration \
// RUN:   -Rcompile-job-cache-hit -emit-obj -o %t/output.o \
// RUN:   -serialize-diagnostic-file %t/diags %s 2>&1 \
// RUN:   | FileCheck %s --allow-empty --check-prefix=CACHE-MISS

// RUN: c-index-test -read-diagnostics %t/diags 2>&1 | FileCheck %s --check-prefix=SERIALIZED-MISS

// RUN: ls %t/output.o && rm %t/output.o

// RUN: %clang -cc1 -triple x86_64-apple-macos11 -fcas-path %t/cas \
// RUN:   -fcas-fs @%t/casid -fcache-compile-job \
// RUN:   -Wimplicit-function-declaration \
// RUN:   -Wno-error=implicit-function-declaration \
// RUN:   -Rcompile-job-cache-hit -emit-obj -o %t/output.o \
// RUN:   -serialize-diagnostic-file %t/diags %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CACHE-HIT

// RUN: c-index-test -read-diagnostics %t/diags 2>&1 | FileCheck %s --check-prefix=SERIALIZED-HIT

// CACHE-HIT: remark: compile job cache hit
// CACHE-HIT: warning: some warning

// CACHE-MISS: warning: some warning
// CACHE-MISS-NOT: remark: compile job cache hit

// FIXME: serialized diagnostics should include the "compile job cache" remark but without storing
// it in a diagnostics file that we put in the CAS.
// SERIALIZED-HIT: warning: some warning
// SERIALIZED-HIT: Number of diagnostics: 1
// SERIALIZED-MISS: warning: some warning
// SERIALIZED-MISS: Number of diagnostics: 1

// Make sure warnings are merged with driver ones.
// Using normal compilation as baseline.
// RUN: %clang -target x86_64-apple-macos11 -c %s -o %t/t.o -fmodules-cache-path=%t/mcp --serialize-diagnostics %t/t1.diag \
// RUN:   2>&1 | FileCheck %s -check-prefix=WARN
// RUN: env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache \
// RUN:   %clang -target x86_64-apple-macos11 -c %s -o %t/t.o -fmodules-cache-path=%t/mcp --serialize-diagnostics %t/t2.diag \
// RUN:   2>&1 | FileCheck %s -check-prefix=WARN
// RUN: diff %t/t1.diag %t/t2.diag

// Try again with cache hit.
// RUN: rm %t/t2.diag
// RUN: env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache \
// RUN:   %clang -target x86_64-apple-macos11 -c %s -o %t/t.o -fmodules-cache-path=%t/mcp --serialize-diagnostics %t/t2.diag
// RUN: env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache \
// RUN:   %clang -target x86_64-apple-macos11 -c %s -o %t/t.o -fmodules-cache-path=%t/mcp --serialize-diagnostics %t/t2.diag \
// RUN:   2>&1 | FileCheck %s -check-prefix=WARN
// RUN: diff %t/t1.diag %t/t2.diag

// WARN: warning: argument unused during compilation
// WARN: warning: some warning

#warning some warning
