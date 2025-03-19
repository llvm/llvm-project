// clang-format off
/// Test that -mllvm options are forwarded to the remote compiler for DTLTO.

// RUN: rm -rf %t && mkdir %t && cd %t

// RUN: %clang --target=x86_64-linux-gnu %s -shared -flto=thin \
// RUN:   -fthinlto-distributor=%python \
// RUN:   -Xthinlto-distributor=%llvm_src_root/utils/dtlto/local.py \
// RUN:   -fuse-ld=lld \
// RUN:   -nostdlib \
// RUN:   -Werror \
/// Specify -v for both the inital and remote clang invocations.
// RUN:   -v \
// RUN:   -Wl,-mllvm=-thinlto-remote-compiler-arg=-v \
/// -thinlto-remote-compiler-arg is a safe -mllvm option to use for this test as
/// it is part of the DTLTO implemenation.
// RUN:   -mllvm -thinlto-remote-compiler-arg=llvm1 \
// RUN:   -mllvm=-thinlto-remote-compiler-arg=llvm2 \
// RUN:   2>&1 | FileCheck %s

// -mllvm arguments are forwarded to `clang -cc1`.
// CHECK:       -mllvm -thinlto-remote-compiler-arg=llvm1
// CHECK-SAME:  -mllvm -thinlto-remote-compiler-arg=llvm2

// -mllvm arguments are forwarded to the remote compiler via lld.
// CHECK:       -mllvm=-thinlto-remote-compiler-arg=-mllvm=-thinlto-remote-compiler-arg=llvm1
// CHECK-SAME:  -mllvm=-thinlto-remote-compiler-arg=-mllvm=-thinlto-remote-compiler-arg=llvm2

// -mllvm arguments are forwarded to `clang -cc1` in the remote execution.
// CHECK:      -mllvm -thinlto-remote-compiler-arg=llvm1
// CHECK-SAME: -mllvm -thinlto-remote-compiler-arg=llvm2

int _start() { return 0; }
