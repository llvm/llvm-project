// REQUIRES: lld

/// Check DTLTO options are forwarded to the linker.

// RUN: echo "--target=x86_64-linux-gnu \
// RUN:   -Xthinlto-distributor=a1 \
// RUN:   -Xthinlto-distributor=a2,a3 \
// RUN:   -fuse-ld=lld" > %t.rsp

/// Check that options are forwarded as expected with --thinlto-distributor=.
// RUN: %clang -### @%t.rsp -fthinlto-distributor=d.exe %s 2>&1 | \
// RUN:   FileCheck %s --implicit-check-not=warning --check-prefix=FORWARD

// FORWARD: ld.lld
// FORWARD-SAME: "--thinlto-distributor=d.exe"
// FORWARD-SAME: "--thinlto-remote-compiler={{.*}}clang
// FORWARD-SAME: "--thinlto-distributor-arg=a1"
// FORWARD-SAME: "--thinlto-distributor-arg=a2"
// FORWARD-SAME: "--thinlto-distributor-arg=a3"


/// Check that options are not added without --thinlto-distributor= and
/// that there is an unused option warning issued for -Xthinlto-distributor=
/// options. We specify -flto here as these options should be unaffected by it.
// RUN: %clang -### @%t.rsp -flto=thin %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=NODIST,NOMORE --implicit-check-not=warning

// NODIST: warning: argument unused during compilation: '-Xthinlto-distributor=a1'
// NODIST: warning: argument unused during compilation: '-Xthinlto-distributor=a2,a3'
// NODIST: ld.lld
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
