// REQUIRES: x86-registered-target
// REQUIRES: shell

// RUN: mkdir -p %t; cd %t
// RUN: %clang -target x86_64-unknown-linux-gnu -c -emit-interface-stubs %s -o %t/driver-test3.o
// RUN: llvm-nm %t/driver-test3.o | FileCheck --check-prefix=CHECK-OBJ %s
// RUN: cat %t/driver-test3.ifs | FileCheck --check-prefix=CHECK-IFS %s

// CHECK-OBJ: bar

// CHECK-IFS: --- !experimental-ifs-v2
// CHECK-IFS-NEXT: IfsVersion:
// CHECK-IFS-NEXT: Triple:
// CHECK-IFS-NEXT: ObjectFileFormat:
// CHECK-IFS-NEXT: Symbols:
// CHECK-IFS-NEXT:   - { Name: "bar", Type: Func }
// CHECK-IFS-NEXT: ...

int bar(int a) { return a; }
