// RUN: rm -rf %t && mkdir -p %t

// This compiles twice with replay disabled, ensuring that we get the same outputs for the same key.

// RUN: env LLVM_CACHE_CAS_PATH=%t/cas CLANG_CACHE_TEST_DETERMINISTIC_OUTPUTS=1 %clang-cache \
// RUN:   %clang -target x86_64-apple-macos11 -c %s -o %t/t.o -Rcompile-job-cache 2> %t/out.txt
// RUN: FileCheck %s --check-prefix=CACHE-MISS --input-file=%t/out.txt

// CACHE-MISS: remark: compile job cache miss
// CACHE-MISS: remark: compile job cache miss
