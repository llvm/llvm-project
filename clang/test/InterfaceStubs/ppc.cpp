// REQUIRES: powerpc-registered-target

// RUN: %clang -x c++ -target powerpc64le-unknown-linux-gnu -o - %s \
// RUN:   -emit-interface-stubs -emit-merged-ifs -S | \
// RUN: FileCheck -check-prefix=CHECK-IFS %s

// CHECK-IFS: --- !ifs-v1
// CHECK-IFS: IfsVersion: 3.0
// CHECK-IFS: Target: powerpc64le
// CHECK-IFS: Symbols:
// CHECK-IFS:   - { Name: _Z8helloPPCv, Type: Func }
// CHECK-IFS: ...

int helloPPC();