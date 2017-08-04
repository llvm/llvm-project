// RUN: %clang_cc1 -pg -mfentry -triple i386-unknown-unknown -emit-llvm -o - %s | FileCheck  %s
// RUN: %clang_cc1 -pg -mfentry -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -mfentry -triple i386-unknown-unknown -emit-llvm -o - %s | FileCheck -check-prefix=NOPG %s
// RUN: %clang_cc1 -mfentry -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck -check-prefix=NOPG %s

int foo(void) {
  return 0;
}

//CHECK: attributes #{{[0-9]+}} = { {{.*}}"fentry-call"="true"{{.*}} }
//NOPG-NOT: attributes #{{[0-9]+}} = { {{.*}}"fentry-call"{{.*}} }
