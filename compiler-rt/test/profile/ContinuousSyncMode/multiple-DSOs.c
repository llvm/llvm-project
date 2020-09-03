// REQUIRES: darwin

// RUN: echo "void dso1(void) {}" > %t.dso1.c
// RUN: echo "void dso2(void) {}" > %t.dso2.c
// RUN: %clang_pgogen -dynamiclib -o %t.dso1.dylib %t.dso1.c
// RUN: %clang_pgogen -dynamiclib -o %t.dso2.dylib %t.dso2.c
// RUN: %clang_pgogen -o %t.exe %s %t.dso1.dylib %t.dso2.dylib
// RUN: env LLVM_PROFILE_FILE="%c%t.profraw" %run %t.exe
// RUN: llvm-profdata show --counts --all-functions %t.profraw | FileCheck %s

// CHECK-LABEL: Counters:
// CHECK-NEXT:   dso1:
// CHECK-NEXT:     Hash: 0x{{.*}}
// CHECK-NEXT:     Counters: 1
// CHECK-NEXT:     Block counts: [1]
// CHECK-NEXT:   dso2:
// CHECK-NEXT:     Hash: 0x{{.*}}
// CHECK-NEXT:     Counters: 1
// CHECK-NEXT:     Block counts: [1]
// CHECK-NEXT:   main:
// CHECK-NEXT:     Hash: 0x{{.*}}
// CHECK-NEXT:     Counters: 1
// CHECK-NEXT:     Block counts: [1]
// CHECK-NEXT: Instrumentation level: IR
// CHECK-NEXT: Functions shown: 3
// CHECK-NEXT: Total functions: 3
// CHECK-NEXT: Maximum function count: 1
// CHECK-NEXT: Maximum internal block count: 0

void dso1(void);
void dso2(void);

int main() {
  dso1();
  dso2();
  return 0;
}
