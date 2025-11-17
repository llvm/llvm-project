// Test that emitted indexing data are allowed to "escape" the CAS sandbox without indexing options affecting the CAS key
// (essentially indexing data are produced when the compilation is executed but they are not replayed if the compilation is cached)

// RUN: rm -rf %t && mkdir %t

// RUN: %clang -cc1depscan -fdepscan=inline -o %t/t.rsp -cc1-args \
// RUN:   -cc1 -triple x86_64-apple-macos12 -fcas-path %t/cas -emit-obj %s -o %t/t.o

// RUN: %clang @%t/t.rsp -Rcompile-job-cache -index-store-path %t/idx -index-unit-output-path t.o 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CACHE-MISS
// RUN: find %t/idx/*/records | count 1

// RUN: rm -rf %t/idx && mkdir %t/idx
// RUN: %clang @%t/t.rsp -Rcompile-job-cache 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CACHE-HIT
// RUN: ls %t/idx | count 0

// CACHE-MISS: remark: compile job cache miss
// CACHE-HIT: remark: compile job cache hit
