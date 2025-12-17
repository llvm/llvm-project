// REQUIRES: x86-registered-target

// RUN: mkdir -p %t; cd %t
// RUN: %clang -target x86_64-unknown-linux-gnu -c -emit-interface-stubs %s -o %t/driver-test3.o
// RUN: llvm-nm %t/driver-test3.o | FileCheck --check-prefix=CHECK-OBJ %s
// RUN: cat %t/driver-test3.ifs | FileCheck --check-prefix=CHECK-IFS %s

// CHECK-OBJ: bar

// CHECK-IFS: --- !ifs-v1
// CHECK-IFS-NEXT: IfsVersion:
// CHECK-IFS-NEXT: Target:
// CHECK-IFS-NEXT: Symbols:
// CHECK-IFS-NEXT:   - { Name: "bar", Type: Func }
// CHECK-IFS-NEXT: ...

int bar(int a) { return a; }
