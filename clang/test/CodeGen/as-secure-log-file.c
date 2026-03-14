// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-obj %s -o %t.o -as-secure-log-file %t.log
// RUN: FileCheck %s -input-file %t.log
// CHECK: "foobar"

void test(void) {
  __asm__(".secure_log_unique \"foobar\"");
}
