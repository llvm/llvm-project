// RUN: rm -rf %t && mkdir -p %t

// RUN: %clang -cc1depscan -fdepscan=inline -o %t.rsp -cc1-args \
// RUN:   -cc1 -triple x86_64-apple-macos11 \
// RUN:   -fcas-path %t/cas -emit-obj %s -o %t/output.o
// RUN: %clang @%t.rsp -fcache-compile-job -Rcompile-job-cache 2>&1 \
// RUN:   | FileCheck %s --allow-empty --check-prefix=CACHE-MISS
// RUN: ls %t/output.o && rm %t/output.o
// RUN: %clang -cc1depscan -fdepscan=inline -o %t.rsp -cc1-args \
// RUN:   -cc1 -triple x86_64-apple-macos11 \
// RUN:   -fcas-path %t/cas -emit-obj %s -o %t/output.o
// RUN: %clang @%t.rsp -fcache-compile-job -Rcompile-job-cache 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CACHE-HIT
// RUN: ls %t/output.o && rm %t/output.o
// RUN: cd %t
// RUN: %clang -cc1depscan -fdepscan=inline -o %t.rsp -cc1-args \
// RUN:   -cc1 -triple x86_64-apple-macos11 \
// RUN:   -fcas-path %t/cas -emit-obj %s -o output.o
// RUN: %clang @%t.rsp -fcache-compile-job -Rcompile-job-cache 2>&1 \
// RUN:   | FileCheck %s --allow-empty --check-prefix=CACHE-HIT
// RUN: ls %t/output.o
//
// Check for a cache hit if the CAS moves:
// RUN: mv %t/cas %t/cas.moved
// RUN: %clang -cc1depscan -fdepscan=inline -o %t.rsp -cc1-args \
// RUN:   -cc1 -triple x86_64-apple-macos11 \
// RUN:   -fcas-path %t/cas.moved -emit-obj %s -o output.o
// RUN: %clang @%t.rsp -fcache-compile-job -Rcompile-job-cache 2> %t/cache-hit.out
// RUN: FileCheck %s -input-file=%t/cache-hit.out --check-prefix=CACHE-HIT
// RUN: ls %t/output.o

// RUN: cat %t/cache-hit.out | sed \
// RUN:   -e "s/^.*hit for '//" \
// RUN:   -e "s/' .*$//" > %t/cache-key
// RUN: cat %t/cache-hit.out | sed \
// RUN:   -e "s/^.*=> '//" \
// RUN:   -e "s/' .*$//" > %t/cache-result

// Check for a handling error if the result in the CAS is removed but not action cache.
// First need to construct the include-tree and the compilation key.
// RUN: %clang -cc1depscan -fdepscan=inline -o %t.rsp -cc1-args \
// RUN:   -cc1 -triple x86_64-apple-macos11 \
// RUN:   -fcas-path %t/cas.new -emit-obj %s -o output.o
// Add the 'key => result' association we got earlier.
// RUN: llvm-cas --cas %t/cas.new --put-cache-key @%t/cache-key @%t/cache-result
// RUN: %clang @%t.rsp -fcache-compile-job -Rcompile-job-cache 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CACHE-RESULT
// RUN: ls %t/output.o
//
// CACHE-HIT: remark: compile job cache hit
// CACHE-MISS-NOT: remark: compile job cache hit
// CACHE-RESULT: remark: compile job cache miss
// CACHE-RESULT-SAME: result not found
