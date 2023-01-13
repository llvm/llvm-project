// REQUIRES: remote-cache-service

// Need a short path for the unix domain socket (and unique for this test file).
// RUN: rm -f %{remote-cache-dir}/%basename_t
// RUN: rm -rf %t && mkdir -p %t

// Baseline to check we got expected outputs.
// RUN: %clang -target x86_64-apple-macos11 -c %s -o %t/t.o -MMD -MT dependencies -MF %t/t.d --serialize-diagnostics %t/t.dia
// Adding `LLBUILD_TASK_ID` just to make sure there's no failure if that is set but `LLBUILD_CONTROL_FD` is not.
// RUN: llvm-remote-cache-test -socket-path=%{remote-cache-dir}/%basename_t -cache-path=%t/cache -- env LLBUILD_TASK_ID=1 LLVM_CACHE_CAS_PATH=%t/cas %clang-cache \
// RUN:   %clang -target x86_64-apple-macos11 -c %s -o %t/t1.o -MMD -MT dependencies -MF %t/t1.d --serialize-diagnostics %t/t1.dia -Rcompile-job-cache \
// RUN:   2>&1 | FileCheck %s --check-prefix=CACHE-MISS
// RUN: llvm-remote-cache-test -socket-path=%{remote-cache-dir}/%basename_t -cache-path=%t/cache -- env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache \
// RUN:   %clang -target x86_64-apple-macos11 -c %s -o %t/t2.o -MMD -MT dependencies -MF %t/t2.d --serialize-diagnostics %t/t2.dia -Rcompile-job-cache \
// RUN:   2>&1 | FileCheck %s --check-prefix=CACHE-HIT

// CACHE-MISS: remark: compile job cache miss
// CACHE-MISS: warning: some warning

// CACHE-HIT: remark: compile job cache hit
// CACHE-HIT: warning: some warning

// RUN: diff %t/t1.o %t/t2.o
// RUN: diff %t/t.o %t/t1.o

// RUN: diff %t/t1.d %t/t2.d
// RUN: diff %t/t.d %t/t1.d

// RUN: c-index-test -read-diagnostics %t/t1.dia 2>&1 | FileCheck %s --check-prefix=SERIAL_DIAG-MISS --check-prefix=SERIAL_DIAG-COMMON
// RUN: c-index-test -read-diagnostics %t/t2.dia 2>&1 | FileCheck %s --check-prefix=SERIAL_DIAG-HIT --check-prefix=SERIAL_DIAG-COMMON
// SERIAL_DIAG-MISS: warning: compile job cache miss
// SERIAL_DIAG-HIT: warning: compile job cache hit
// SERIAL_DIAG-COMMON: warning: some warning

// Verify the outputs did not go into the on-disk ObjectStore.
// RUN: env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache \
// RUN:   %clang -target x86_64-apple-macos11 -c %s -o %t/t2.o -MMD -MT dependencies -MF %t/t2.d --serialize-diagnostics %t/t2.dia -Rcompile-job-cache \
// RUN:   2>&1 | FileCheck %s --check-prefix=CACHE-MISS

#warning some warning
void test() {}
