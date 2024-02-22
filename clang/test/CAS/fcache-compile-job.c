// RUN: rm -rf %t && mkdir -p %t
// RUN: llvm-cas --cas %t/cas --ingest %s > %t/casid
//
// RUN: %clang -cc1 -triple x86_64-apple-macos11 \
// RUN:   -fcas-path %t/cas -fcas-fs @%t/casid -fcache-compile-job \
// RUN:   -Rcompile-job-cache -emit-obj %s -o %t/output.o 2>&1 \
// RUN:   | FileCheck %s --allow-empty --check-prefix=CACHE-MISS
// RUN: ls %t/output.o && rm %t/output.o
// RUN: %clang -cc1 -triple x86_64-apple-macos11 \
// RUN:   -fcas-path %t/cas -fcas-fs @%t/casid -fcache-compile-job \
// RUN:   -Rcompile-job-cache -emit-obj %s -o %t/output.o 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CACHE-HIT
// RUN: ls %t/output.o && rm %t/output.o
// RUN: cd %t
// RUN: %clang -cc1 -triple x86_64-apple-macos11 \
// RUN:   -fcas-path %t/cas -fcas-fs @%t/casid -fcache-compile-job \
// RUN:   -Rcompile-job-cache -emit-obj %s -o output.o 2>&1 \
// RUN:   | FileCheck %s --allow-empty --check-prefix=CACHE-HIT
// RUN: ls %t/output.o
//
// Check for a cache hit if the CAS moves:
// RUN: mv %t/cas %t/cas.moved
// RUN: %clang -cc1 -triple x86_64-apple-macos11 \
// RUN:   -fcas-path %t/cas.moved -fcas-fs @%t/casid -fcache-compile-job \
// RUN:   -Rcompile-job-cache -emit-obj %s -o output.o 2> %t/cache-hit.out
// RUN: FileCheck %s -input-file=%t/cache-hit.out --check-prefix=CACHE-HIT
// RUN: ls %t/output.o

// RUN: cat %t/cache-hit.out | sed \
// RUN:   -e "s/^.*hit for '//" \
// RUN:   -e "s/' .*$//" > %t/cache-key
// RUN: cat %t/cache-hit.out | sed \
// RUN:   -e "s/^.*=> '//" \
// RUN:   -e "s/' .*$//" > %t/cache-result

// Check for a handling error if the CAS is removed but not action cache.
// First need to ingest the input file so the compile cache can be constructed.
// RUN: llvm-cas --ingest --cas %t/cas.new  %s
// Add the 'key => result' association we got earlier.
// RUN: llvm-cas --cas %t/cas.new --put-cache-key @%t/cache-key @%t/cache-result
// RUN: %clang -cc1 -triple x86_64-apple-macos11 \
// RUN:   -fcas-path %t/cas.new -fcas-fs @%t/casid -fcache-compile-job \
// RUN:   -Rcompile-job-cache -emit-obj %s -o output.o 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CACHE-RESULT
// RUN: ls %t/output.o
//
// CACHE-HIT: remark: compile job cache hit
// CACHE-MISS-NOT: remark: compile job cache hit
// CACHE-RESULT: remark: compile job cache miss
// CACHE-RESULT-SAME: result not found
