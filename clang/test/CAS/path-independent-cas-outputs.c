// RUN: rm -rf %t && mkdir -p %t/a %t/b

// Check that we got a cache hit even though the output paths are different.

// RUN: env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache \
// RUN:   %clang -target x86_64-apple-macos11 -c %s -o %t/a/t1.o -MMD -MT dependencies -MF %t/a/t1.d --serialize-diagnostics %t/a/t1.dia -Rcompile-job-cache \
// RUN:   2>&1 | FileCheck %s --check-prefix=CACHE-MISS
// RUN: env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache \
// RUN:   %clang -target x86_64-apple-macos11 -c %s -o %t/b/t2.o -MMD -MT dependencies -MF %t/b/t2.d --serialize-diagnostics %t/b/t2.dia -Rcompile-job-cache \
// RUN:   2>&1 | FileCheck %s --check-prefix=CACHE-HIT

// Check that output path is correctly detected with -working-directory.

// RUN: env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache \
// RUN:   %clang -target x86_64-apple-macos11 -c %s -working-directory %t -o a/t1_working_dir.o -DNEW_FLAG -Rcompile-job-cache \
// RUN:   2>&1 | FileCheck %s --check-prefix=CACHE-MISS
// RUN: env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache \
// RUN:   %clang -target x86_64-apple-macos11 -c %s -working-directory %t -o b/t2_working_dir.o -DNEW_FLAG -Rcompile-job-cache \
// RUN:   2>&1 | FileCheck %s --check-prefix=CACHE-HIT
// RUN: diff %t/a/t1_working_dir.o %t/b/t2_working_dir.o

// Check that output path is correctly detected with - (stdout)

// RUN: env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache \
// RUN:   %clang -target x86_64-apple-macos11 -c %s -o - -DNEW_FLAG2 -Rcompile-job-cache > %t/a/t1_stdout.o
// RUN: env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache \
// RUN:   %clang -target x86_64-apple-macos11 -c %s -o %t/b/t2_stdout.o -DNEW_FLAG2 -Rcompile-job-cache \
// RUN:   2>&1 | FileCheck %s --check-prefix=CACHE-HIT
// RUN: diff %t/a/t1_stdout.o %t/b/t2_stdout.o

// Check PCH output

// RUN: env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache \
// RUN:   %clang -target x86_64-apple-macos11 -x c-header %s -o %t/a/t1.pch -Rcompile-job-cache 2>&1 | FileCheck %s --check-prefix=CACHE-MISS
// RUN: env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache \
// RUN:   %clang -target x86_64-apple-macos11 -x c-header %s -o %t/b/t2.pch -Rcompile-job-cache 2>&1 | FileCheck %s --check-prefix=CACHE-HIT

// CACHE-MISS: remark: compile job cache miss
// CACHE-HIT: remark: compile job cache hit

// Repeat to diff outputs produced from each invocation. CAS path is different to avoid cache hits.

// RUN: rm -rf %t && mkdir -p %t

// Baseline to check we got expected outputs.
// RUN: %clang -target x86_64-apple-macos11 -c %s -o %t/t.o -MMD -MT dependencies -MF %t/t.d --serialize-diagnostics %t/t.dia
// RUN: env LLVM_CACHE_CAS_PATH=%t/a/cas %clang-cache \
// RUN:   %clang -target x86_64-apple-macos11 -c %s -o %t/a/t1.o -MMD -MT dependencies -MF %t/a/t1.d --serialize-diagnostics %t/a/t1.dia
// RUN: env LLVM_CACHE_CAS_PATH=%t/b/cas %clang-cache \
// RUN:   %clang -target x86_64-apple-macos11 -c %s -o %t/b/t2.o -MMD -MT dependencies -MF %t/b/t2.d --serialize-diagnostics %t/b/t2.dia

// RUN: diff %t/a/t1.o %t/b/t2.o
// RUN: diff %t/t.o %t/a/t1.o

// RUN: diff %t/a/t1.dia %t/b/t2.dia
// RUN: diff %t/t.dia %t/a/t1.dia

// RUN: diff %t/a/t1.d %t/b/t2.d
// RUN: diff %t/t.d %t/a/t1.d

// RUN: env LLVM_CACHE_CAS_PATH=%t/a/cas %clang-cache \
// RUN:   %clang -target x86_64-apple-macos11 -x c-header %s -o %t/a/t1.pch
// RUN: env LLVM_CACHE_CAS_PATH=%t/b/cas %clang-cache \
// RUN:   %clang -target x86_64-apple-macos11 -x c-header %s -o %t/b/t2.pch

// RUN: diff %t/a/t1.pch %t/b/t2.pch

// Check that caching is independent of whether '--serialize-diagnostics' exists or not.

// Check with the option missing then present.
// RUN: env LLVM_CACHE_CAS_PATH=%t/d1/cas %clang-cache \
// RUN:   %clang -target x86_64-apple-macos11 -c %s -o %t/t1.o -Rcompile-job-cache \
// RUN:   2>&1 | FileCheck %s --check-prefix=CACHE-MISS
// RUN: env LLVM_CACHE_CAS_PATH=%t/d1/cas %clang-cache \
// RUN:   %clang -target x86_64-apple-macos11 -c %s -o %t/t2.o --serialize-diagnostics %t/t1.dia -Rcompile-job-cache \
// RUN:   2>&1 | FileCheck %s --check-prefix=CACHE-HIT

// Check with the option present then missing.
// RUN: env LLVM_CACHE_CAS_PATH=%t/d2/cas %clang-cache \
// RUN:   %clang -target x86_64-apple-macos11 -c %s -o %t/t1.o --serialize-diagnostics %t/t2.dia -Rcompile-job-cache \
// RUN:   2>&1 | FileCheck %s --check-prefix=CACHE-MISS
// RUN: env LLVM_CACHE_CAS_PATH=%t/d2/cas %clang-cache \
// RUN:   %clang -target x86_64-apple-macos11 -c %s -o %t/t2.o -Rcompile-job-cache \
// RUN:   2>&1 | FileCheck %s --check-prefix=CACHE-HIT

// RUN: c-index-test -read-diagnostics %t/t1.dia 2>&1 | FileCheck %s --check-prefix=SERIAL_DIAG-HIT --check-prefix=SERIAL_DIAG-COMMON
// RUN: c-index-test -read-diagnostics %t/t2.dia 2>&1 | FileCheck %s --check-prefix=SERIAL_DIAG-MISS --check-prefix=SERIAL_DIAG-COMMON
// SERIAL_DIAG-MISS: warning: compile job cache miss
// SERIAL_DIAG-HIT: warning: compile job cache hit
// SERIAL_DIAG-COMMON: warning: some warning

#warning some warning
void test() {}
