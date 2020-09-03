// REQUIRES: powerpc-registered-target

// RUN: %clang -x c++ -target powerpc64le-unknown-linux-gnu -o - %s \
// RUN:   -emit-interface-stubs -emit-merged-ifs -S | \
// RUN: FileCheck -check-prefix=CHECK-IFS %s

// CHECK-IFS: --- !experimental-ifs-v2
// CHECK-IFS: IfsVersion: 2.0
// CHECK-IFS: Triple: powerpc64le
// CHECK-IFS: Symbols:
// CHECK-IFS:   - { Name: _Z8helloPPCv, Type: Func }
// CHECK-IFS: ...

int helloPPC();
