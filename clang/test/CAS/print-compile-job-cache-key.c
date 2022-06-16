// REQUIRES: shell

// RUN: rm -rf %t && mkdir -p %t
// RUN: llvm-cas --cas %t/cas --ingest --data %s > %t/casid
//
// RUN: %clang -cc1 -triple x86_64-apple-macos11 \
// RUN:   -fcas-path %t/cas -fcas-fs @%t/casid -fcache-compile-job \
// RUN:   -Rcompile-job-cache-miss -emit-obj -o %t/output.o %s 2> %t/output.txt
//
// RUN: cat %t/output.txt | sed \
// RUN:   -e "s/^.*miss for '//" \
// RUN:   -e "s/' .*$//" > %t/cache-key
//
// RUN: not clang-cas-test -print-compile-job-cache-key -cas %t/cas 2>&1 | FileCheck %s -check-prefix=NO_KEY
// NO_KEY: missing compile-job cache key
//
// RUN: not clang-cas-test -print-compile-job-cache-key -cas %t/cas asdf 2>&1 | FileCheck %s -check-prefix=INVALID_KEY
// INVALID_KEY: invalid cas-id 'asdf'
//
// RUN: not clang-cas-test -print-compile-job-cache-key -cas %t/cas @%t/casid 2>&1 | FileCheck %s -check-prefix=NOT_A_KEY
// NOT_A_KEY: not a valid cache key
//
// RUN: clang-cas-test -print-compile-job-cache-key -cas %t/cas @%t/cache-key | FileCheck %s
//
// CHECK: command-line: llvmcas://
// CHECK:   -cc1
// CHECK:   -fcas-path llvm.cas.builtin.v2[BLAKE3]
// CHECK:   -fcas-fs llvmcas://
// CHECK:   -x c {{.*}}print-compile-job-cache-key.c
// CHECK: computation: llvmcas://
// CHECK:   -cc1
// CHECK: filesystem: llvmcas://
// CHECK:   file llvmcas://
// CHECK: version: llvmcas://
// CHECK:   clang version
