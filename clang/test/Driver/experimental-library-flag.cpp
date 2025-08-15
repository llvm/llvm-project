// On some platforms, -stdlib=libc++ is currently ignored, so -lc++experimental is not added.
// Once -stdlib=libc++ works on those, this XFAIL can be removed.
// XFAIL: target={{.*-windows-msvc.*}}, target={{.*-(ps4|ps5)}}

// For some reason, this fails with a core dump on AIX. This needs to be investigated.
// UNSUPPORTED: target={{.*}}-aix{{.*}}

// RUN: %clangxx -fexperimental-library -stdlib=libc++ -### %s 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-LIBCXX %s
// RUN: %clangxx -fexperimental-library -stdlib=libstdc++ -### %s 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-LIBSTDCXX %s
// RUN: %clangxx -fexperimental-library -stdlib=libc++ -nostdlib++ -### %s 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-NOSTDLIB %s

/// The FreeBSD driver did not support -stdlib=libstdc++ previously, check that it does the right thing here.
// RUN: %clangxx --target=x86_64-unknown-freebsd -fexperimental-library -stdlib=libc++ -### %s 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-LIBCXX %s
// RUN: %clangxx --target=x86_64-unknown-freebsd -fexperimental-library -stdlib=libstdc++ -### %s 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-LIBSTDCXX %s
// RUN: %clangxx --target=x86_64-unknown-freebsd -fexperimental-library -stdlib=libc++ -nostdlib++ -### %s 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-NOSTDLIB %s

// -fexperimental-library must be passed to CC1.
// CHECK: -fexperimental-library

// Depending on the stdlib in use, we should (or not) pass -lc++experimental.
// CHECK-LIBCXX: -lc++experimental
// CHECK-LIBSTDCXX-NOT: -lc++experimental
// CHECK-NOSTDLIB-NOT: -lc++experimental
