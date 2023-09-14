// Ensure MCTargetOptionsCommandFlags are parsable under -mllvm
// RUN: %clang -mllvm --help -c %s -o /dev/null | FileCheck %s
// CHECK: --asm-show-inst
// CHECK: --dwarf-version
// CHECK: --dwarf64
// CHECK: --emit-dwarf-unwind
// CHECK: --fatal-warnings
// CHECK: --incremental-linker-compatible
// CHECK: --mc-relax-all
// CHECK: --no-deprecated-warn
// CHECK: --no-type-check
// CHECK: --no-warn
