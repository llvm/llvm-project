/// Check DTLTO options are forwarded to the linker.

// REQUIRES: lld

// RUN: echo "--target=x86_64-linux-gnu \
// RUN:   -Xthinlto-distributor=distarg1 \
// RUN:   -Xthinlto-distributor=distarg2,distarg3 \
// RUN:   -mllvm llvm1 \
// RUN:   -mllvm=llvm2 \
// RUN:   -fuse-ld=lld" > %t.rsp


/// Check that options are forwarded as expected with --thinlto-distributor=.
// RUN: %clang -### @%t.rsp -fthinlto-distributor=dist.exe %s 2>&1 | \
// RUN:   FileCheck %s --implicit-check-not=warning

// CHECK: ld.lld
// CHECK-SAME: "--thinlto-distributor=dist.exe"
// CHECK-SAME: "-mllvm=-thinlto-remote-compiler={{.*}}clang
// CHECK-SAME: "-mllvm=-thinlto-distributor-arg=distarg1"
// CHECK-SAME: "-mllvm=-thinlto-distributor-arg=distarg2"
// CHECK-SAME: "-mllvm=-thinlto-distributor-arg=distarg3"
// CHECK-SAME: "-mllvm=-thinlto-remote-compiler-arg=-mllvm=llvm1"
// CHECK-SAME: "-mllvm=-thinlto-remote-compiler-arg=-mllvm=llvm2"


/// Check that options are not added without --thinlto-distributor= and
/// that there is an unused option warning issued for -Xthinlto-distributor= options. We
/// specify -flto here as these options should be unaffected by it.
// RUN: %clang -### @%t.rsp -flto=thin %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=NONE,NOMORE --implicit-check-not=warning

// NONE: warning: argument unused during compilation: '-Xthinlto-distributor=distarg1'
// NONE: warning: argument unused during compilation: '-Xthinlto-distributor=distarg2,distarg3'
// NONE:     ld.lld
// NOMORE-NOT: --thinlto-distributor=
// NOMORE-NOT: -thinlto-remote-compiler=
// NOMORE-NOT: -mllvm
// NOMORE-NOT: -thinlto-distributor-arg=


/// Check the expected arguments are forwarded by default with only
/// --thinlto-distributor=.
// RUN: %clang --target=x86_64-linux-gnu -fthinlto-distributor=dist.exe \
// RUN:   -fuse-ld=lld -Werror -### %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=DEFAULT,NOMORE

// DEFAULT: ld.lld
// DEFAULT-SAME: "--thinlto-distributor=dist.exe"
// DEFAULT-SAME: "-mllvm=-thinlto-remote-compiler={{.*}}clang
