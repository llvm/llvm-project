// RUN: %clang_cc1 -pg -mfentry -triple i386-unknown-unknown -emit-llvm -o - %s | FileCheck  %s
// RUN: %clang_cc1 -pg -mfentry -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -mfentry -triple i386-unknown-unknown -emit-llvm -o - %s | FileCheck -check-prefix=NOPG %s
// RUN: %clang_cc1 -mfentry -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck -check-prefix=NOPG %s

int foo(void) {
  return 0;
}

int __attribute__((no_instrument_function)) no_instrument(void) {
  return foo();
}

//CHECK: attributes #0 = { {{.*}}"fentry-call"="true"{{.*}} }
//CHECK: attributes #1 = { {{.*}} }
//CHECK-NOT: attributes #1 = { {{.*}}"fentry-call"="true"{{.*}} }
//NOPG-NOT: attributes #0 = { {{.*}}"fentry-call"{{.*}} }
//NOPG-NOT: attributes #1 = { {{.*}}"fentry-call"{{.*}} }
