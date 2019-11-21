// REQUIRES: powerpc-registered-target

// RUN: %clang -x c++ -target powerpc64le-unknown-linux-gnu -o - %s \
// RUN:   -emit-interface-stubs -emit-merged-ifs -S | \
// RUN: FileCheck -check-prefix=CHECK-IFS %s

 // CHECK-IFS: --- !experimental-ifs-v1
 // CHECK-IFS: IfsVersion:      1.0
 // CHECK-IFS: Triple: powerpc64le
 // CHECK-IFS: Symbols:
 // CHECK-IFS:   _Z8helloPPCv: { Type: Func }
 // CHECK-IFS: ...

int helloPPC();
