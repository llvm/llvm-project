// REQUIRES: lld

/// Check DTLTO options are forwarded to the linker.

// RUN: echo "--target=x86_64-linux-gnu \
// RUN:   -Xthinlto-distributor=a1 \
// RUN:   -Xthinlto-distributor=a2,a3 \
// RUN:   -fuse-ld=lld" > %t.rsp

/// Check that options are forwarded as expected with --thinlto-distributor=.
// RUN: %clang -### @%t.rsp -fthinlto-distributor=d.exe %s 2>&1 | \
// RUN:   FileCheck %s --implicit-check-not=warning

// CHECK: ld.lld
// CHECK-SAME: "--thinlto-distributor=d.exe"
// CHECK-SAME: "--thinlto-remote-compiler={{.*}}clang
// CHECK-SAME: "--thinlto-distributor-arg=a1"
// CHECK-SAME: "--thinlto-distributor-arg=a2"
// CHECK-SAME: "--thinlto-distributor-arg=a3"


/// Check that options are not added without --thinlto-distributor= and
/// that there is an unused option warning issued for -Xthinlto-distributor=
/// options. We specify -flto here as these options should be unaffected by it.
// RUN: %clang -### @%t.rsp -flto=thin %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=NONE,NOMORE --implicit-check-not=warning

// NONE: warning: argument unused during compilation: '-Xthinlto-distributor=a1'
// NONE: warning: argument unused during compilation: '-Xthinlto-distributor=a2,a3'
// NONE:     ld.lld
// NOMORE-NOT: distributor
// NOMORE-NOT: remote-compiler


/// Check the expected arguments are forwarded by default with only
/// --thinlto-distributor=.
// RUN: %clang --target=x86_64-linux-gnu -fthinlto-distributor=d.exe \
// RUN:   -fuse-ld=lld -Werror -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEFAULT,NOMORE

// DEFAULT: ld.lld
// DEFAULT-SAME: "--thinlto-distributor=d.exe"
// DEFAULT-SAME: "--thinlto-remote-compiler={{.*}}clang
