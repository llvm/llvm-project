// REQUIRES: powerpc-registered-target
// RUN: %clang -### --driver-mode=g++ -target powerpc64le-linux-gnu %s \
// RUN:  --gcc-toolchain=%S/Inputs/powerpc64le-linux-gnu-tree/gcc-11.2.0 \
// RUN:  -mabi=ieeelongdouble -stdlib=libstdc++ 2>&1 | FileCheck %s
// RUN: %clang -### --driver-mode=g++ -target powerpc64le-linux-gnu %s \
// RUN:  -mabi=ieeelongdouble -stdlib=libc++ 2>&1 | FileCheck %s
// RUN: %clang -### --driver-mode=g++ -target powerpc64le-linux-gnu %s\
// RUN:  -mabi=ieeelongdouble -stdlib=libc++ -Wno-unsupported-abi 2>&1 | \
// RUN:  FileCheck %s --check-prefix=NOWARN

// CHECK: warning: float ABI 'ieeelongdouble' is not supported by current library
// NOWARN-NOT: warning: float ABI 'ieeelongdouble' is not supported by current library
long double foo(long double x) { return x; }
