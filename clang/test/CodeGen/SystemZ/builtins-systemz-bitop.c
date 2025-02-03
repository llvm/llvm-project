// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -target-cpu arch15 -triple s390x-ibm-linux -Wall -Wno-unused -Werror -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -target-cpu arch15 -triple s390x-ibm-linux -Wall -Wno-unused -Werror -emit-llvm -x c++ %s -o - | FileCheck %s

unsigned long test_bdepg(unsigned long a, unsigned long b) {
// CHECK-LABEL: test_bdepg
// CHECK: call i64 @llvm.s390.bdepg(i64 {{.*}}, i64 {{.*}})
  return __builtin_s390_bdepg(a, b);
}

unsigned long test_bextg(unsigned long a, unsigned long b) {
// CHECK-LABEL: test_bextg
// CHECK: call i64 @llvm.s390.bextg(i64 {{.*}}, i64 {{.*}})
  return __builtin_s390_bextg(a, b);
}

