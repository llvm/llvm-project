// Ensure both the first clang process and the daemon have logging enabled.
// It's hard to check this exhaustively, but in practice if the daemon does not
// enable logging there are currently zero records in the log.

// REQUIRES: system-darwin, clang-cc1daemon

// RUN: rm -rf %t && mkdir %t
// RUN: env LLVM_CACHE_CAS_PATH=%t/cas LLVM_CAS_LOG=1 LLVM_CAS_DISABLE_VALIDATION=1 %clang \
// RUN:   -cc1depscan -fdepscan=daemon -fdepscan-include-tree -o - \
// RUN:   -cc1-args -cc1 -triple x86_64-apple-macosx11.0.0 -emit-obj %s -o %t/t.o -fcas-path %t/cas
// RUN: FileCheck %s --input-file %t/cas/v1.log

// CHECK: [[PID1:[0-9]*]] {{[0-9]*}}: mmap '{{.*}}v8.index'
// CHECK: [[PID1]] {{[0-9]*}}: create subtrie

// CHECK: [[PID2:[0-9]*]] {{[0-9]*}}: mmap '{{.*}}v8.index'
// Even a minimal compilation involves at least 9 records for the cache key.
// CHECK-COUNT-9: [[PID2]] {{[0-9]*}}: create record

// CHECK: [[PID1]] {{[0-9]*}}: close mmap '{{.*}}v8.index'
