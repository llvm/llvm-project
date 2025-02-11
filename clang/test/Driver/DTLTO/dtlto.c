/// Check DTLTO options are forwarded to the linker.

// REQUIRES: lld

// RUN: echo "-target x86_64-linux-gnu \
// RUN:   -Xdist distarg1 \
// RUN:   -Xdist distarg2 \
// RUN:   -fuse-ld=lld" > %t.rsp


/// Check that options are forwarded as expected with --thinlto-distributor=.
// RUN: %clang -### @%t.rsp -fthinlto-distributor=dist.exe %s 2>&1 | \
// RUN:   FileCheck %s --implicit-check-not=warning

// CHECK: ld.lld
// CHECK-SAME: "--thinlto-distributor=dist.exe"
// CHECK-SAME: "--thinlto-remote-opt-tool={{.*}}clang
// CHECK-SAME: "-mllvm=-thinlto-distributor-arg=distarg1"
// CHECK-SAME: "-mllvm=-thinlto-distributor-arg=distarg2"


/// Check that options are not added without --thinlto-distributor= and
/// that there is an unused option warning issued for -Xdist options. We
/// specify -flto here as these options should be unaffected by it.
// RUN: %clang -### @%t.rsp -flto=thin %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=NONE,NOMORE --implicit-check-not=warning

// NONE: warning: argument unused during compilation: '-Xdist distarg1'
// NONE: warning: argument unused during compilation: '-Xdist distarg2'
// NONE:     ld.lld
// NOMORE-NOT: --thinlto-distributor=
// NOMORE-NOT: --thinlto-remote-opt-tool=
// NOMORE-NOT: -mllvm
// NOMORE-NOT: -thinlto-distributor-arg=


/// Check the expected arguments are forwarded by default with only
/// --thinlto-distributor=.
// RUN: %clang -### -target x86_64-linux-gnu -fthinlto-distributor=dist.exe -fuse-ld=lld %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEFAULT,NOMORE --implicit-check-not=warning

// DEFAULT: ld.lld
// DEFAULT-SAME: "--thinlto-distributor=dist.exe"
// DEFAULT-SAME: "--thinlto-remote-opt-tool={{.*}}clang
