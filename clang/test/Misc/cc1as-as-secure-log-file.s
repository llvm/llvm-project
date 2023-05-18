// REQUIRES: x86-registered-target

// RUN: %clang -cc1as -triple x86_64-apple-darwin %s -o %t.o -as-secure-log-file %t.log
// RUN: FileCheck %s -input-file %t.log
// CHECK: "foobar"

.secure_log_unique "foobar"
