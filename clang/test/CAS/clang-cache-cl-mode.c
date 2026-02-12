// REQUIRES: ondisk_cas
// REQUIRES: system-windows

// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache %clang_cl -Xclang -Rcompile-job-cache -- %t/test.c 2>&1 | FileCheck %s -check-prefix=CACHE-MISS
// RUN: env LLVM_CACHE_CAS_PATH=%t/cas %clang-cache %clang_cl -Xclang -Rcompile-job-cache -- %t/test.c 2>&1 | FileCheck %s -check-prefix=CACHE-HIT

// CACHE-HIT: remark: compile job cache hit
// CACHE-MISS: remark: compile job cache miss

//--- test.c
int main() { return 0; }
