// REQUIRES: remote-cache-service, shell

// Need a short path for the unix domain socket.
// RUN: CACHE=%{remote-cache-dir}/$(basename %t)
// RUN: rm -rf $CACHE && mkdir -p $CACHE
// RUN: rm -rf %t && mkdir -p %t

// Baseline to check we got expected outputs.
// RUN: %clang -target x86_64-apple-macos11 -c %s -o %t/t.o -MMD -MT dependencies -MF %t/t.d --serialize-diagnostics %t/t.dia
// RUN: llvm-remote-cache-test -cache-path=$CACHE -- env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache \
// RUN:   %clang -target x86_64-apple-macos11 -c %s -o %t/t1.o -MMD -MT dependencies -MF %t/t1.d --serialize-diagnostics %t/t1.dia -Rcompile-job-cache \
// RUN:   2>&1 | FileCheck %s --check-prefix=CACHE-MISS
// RUN: llvm-remote-cache-test -cache-path=$CACHE -- env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache \
// RUN:   %clang -target x86_64-apple-macos11 -c %s -o %t/t2.o -MMD -MT dependencies -MF %t/t2.d --serialize-diagnostics %t/t2.dia -Rcompile-job-cache \
// RUN:   2>&1 | FileCheck %s --check-prefix=CACHE-HIT

// CACHE-MISS: remark: compile job cache miss
// CACHE-MISS: warning: some warning

// CACHE-HIT: remark: compile job cache hit
// CACHE-HIT: warning: some warning

// RUN: diff %t/t1.o %t/t2.o
// RUN: diff %t/t.o %t/t1.o

// RUN: diff %t/t1.dia %t/t2.dia
// RUN: diff %t/t.dia %t/t1.dia

// RUN: diff %t/t1.d %t/t2.d
// RUN: diff %t/t.d %t/t1.d

// Verify the outputs did not go into the on-disk ObjectStore.
// RUN: env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache \
// RUN:   %clang -target x86_64-apple-macos11 -c %s -o %t/t2.o -MMD -MT dependencies -MF %t/t2.d --serialize-diagnostics %t/t2.dia -Rcompile-job-cache \
// RUN:   2>&1 | FileCheck %s --check-prefix=CACHE-MISS

#warning some warning
void test() {}
