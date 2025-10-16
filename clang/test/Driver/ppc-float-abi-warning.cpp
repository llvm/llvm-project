// RUN: %clang -### --driver-mode=g++ -target powerpc64le-linux-gnu %s \
// RUN:  --gcc-toolchain=%S/Inputs/powerpc64le-linux-gnu-tree/gcc-11.2.0 \
// RUN:  -mabi=ieeelongdouble -stdlib=libstdc++ 2>&1 | FileCheck %s
// RUN: %clang -### --driver-mode=g++ -target powerpc64le-linux-gnu %s \
// RUN:  --gcc-toolchain=%S/Inputs/powerpc64le-linux-gnu-tree/gcc-12 \
// RUN:  --dyld-prefix=%S/Inputs/powerpc64le-linux-gnu-tree/gcc-12 \
// RUN:  -mabi=ieeelongdouble -stdlib=libstdc++ 2>&1 | \
// RUN:  FileCheck %s --check-prefix=NOWARN
// RUN: %clang -### --driver-mode=g++ -target powerpc64le-linux-gnu %s\
// RUN:  -stdlib=libc++ 2>&1 | \
// RUN:  FileCheck %s --check-prefix=NOWARN
// RUN: %clang -### --driver-mode=g++ -target powerpc64le-linux-gnu %s\
// RUN:  -mabi=ibmlongdouble -stdlib=libc++ -Wno-unsupported-abi 2>&1 | \
// RUN:  FileCheck %s --check-prefix=NOWARN
// RUN: %clang -### --driver-mode=g++ -target powerpc64le-linux-gnu %s\
// RUN:  -mabi=ieeelongdouble -stdlib=libc++ -Wno-unsupported-abi 2>&1 | \
// RUN:  FileCheck %s --check-prefix=NOWARN
// RUN: %clang -### --driver-mode=g++ -target powerpc64le-linux-gnu %s\
// RUN:  --dyld-prefix=%S/Inputs/powerpc64le-linux-gnu-tree/gcc-12 \
// RUN:  -mabi=%if ppc_linux_default_ieeelongdouble %{ieeelongdouble%} \
// RUN:  %else %{ibmlongdouble%} -stdlib=libc++ 2>&1 | \
// RUN:  FileCheck %s --check-prefix=NOWARN
// RUN: %clang -### --driver-mode=g++ -target powerpc64le-linux-gnu %s\
// RUN:  --dyld-prefix=%S/Inputs/powerpc64le-linux-gnu-tree/gcc-12 \
// RUN:  -mabi=%if ppc_linux_default_ieeelongdouble %{ibmlongdouble%} \
// RUN:  %else %{ieeelongdouble%} -stdlib=libc++ 2>&1 | FileCheck %s

// CHECK: warning: float ABI '{{.*}}' is not supported by current library
// NOWARN-NOT: warning: float ABI '{{.*}}' is not supported by current library
long double foo(long double x) { return x; }
