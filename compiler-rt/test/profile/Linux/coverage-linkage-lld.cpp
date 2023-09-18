// REQUIRES: lld-available

// FIXME: Investigate and fix.
// XFAIL: powerpc64-target-arch

/// With lld --gc-sections we can ensure discarded[01] and their profc/profd
/// variables are discarded.

// RUN: %clang_profgen -fcoverage-mapping -ffunction-sections -fuse-ld=lld -Wl,--gc-sections %S/coverage-linkage.cpp -o %t
// RUN: llvm-nm %t | FileCheck %s

// CHECK-NOT: discarded{{.*}}
