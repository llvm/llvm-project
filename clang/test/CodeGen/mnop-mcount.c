// RUN: %clang_cc1 -pg -mfentry -mnop-mcount -triple s390x-ibm-linux -emit-llvm \
// RUN:   -o - %s 2>&1 | FileCheck  %s
// RUN: not %clang_cc1 -pg -mnop-mcount -triple s390x-ibm-linux -emit-llvm -o - \
// RUN:   %s 2>&1 | FileCheck -check-prefix=NOMFENTRY %s
// RUN: %clang_cc1 -mfentry -mnop-mcount -triple s390x-ibm-linux -emit-llvm -o - \
// RUN:   %s 2>&1 | FileCheck -check-prefix=NOPG %s
// RUN: %clang_cc1 -mnop-mcount -triple s390x-ibm-linux -emit-llvm -o - %s \
// RUN:   2>&1 | FileCheck -check-prefix=NOPG %s

int foo(void) {
  return 0;
}

int __attribute__((no_instrument_function)) no_instrument(void) {
  return foo();
}

//CHECK: attributes #0 = { {{.*}}"mnop-mcount"{{.*}} }
//CHECK: attributes #1 = { {{.*}} }
//CHECK-NOT: attributes #1 = { {{.*}}"mnop-mcount"{{.*}} }
//NOMFENTRY: error: option '-mnop-mcount' cannot be specified without '-mfentry'
//NOPG-NOT: attributes #0 = { {{.*}}"mnop-mcount"{{.*}} }
//NOPG-NOT: attributes #1 = { {{.*}}"mnop-mcount"{{.*}} }
