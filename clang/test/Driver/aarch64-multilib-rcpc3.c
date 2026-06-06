// RUN: %clang --target=aarch64-none-elf -march=armv8.9-a+rcpc3 -print-multi-flags-experimental -multi-lib-config=%S/Inputs/multilib/empty.yaml -c %s 2>&1 | FileCheck %s

// The purpose of this regression test is to make sure that when
// compile options are converted into multilib selection flags, no
// empty strings are accidentally included in the
// -march=armv8.9-a+foo+bar+baz string, leading to two consecutive +
// signs. With +rcpc3 in the input, this used to generate an empty
// string for the anonymous architecture extension corresponding to
// the SubtargetFeature 'rcpc-immo', which is a dependency of rcpc3
// but has no separate extension name for use on command lines. So we
// check that the two named rcpc options appear, and that no ++
// appears before or after.

// CHECK: -march=armv8.9-a
// CHECK-NOT: ++
// CHECK-SAME: +rcpc+rcpc3+
// CHECK-NOT: ++
