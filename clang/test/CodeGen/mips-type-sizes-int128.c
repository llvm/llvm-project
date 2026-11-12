// RUN: not %clang_cc1 -triple mips-none-linux-gnu -emit-llvm -w -o - %s 2> %t1
// RUN: FileCheck --check-prefix=O32 %s < %t1

// RUN: %clang_cc1 -triple mips64-none-linux-gnu -emit-llvm -w -target-abi n32 -o - %s | FileCheck --check-prefix=NEW %s
// RUN: %clang_cc1 -triple mips64-none-linux-gnu -emit-llvm -w -o - %s | FileCheck --check-prefix=NEW %s

// O32 does not support __int128 so it must be tested separately
// N32/N64 behave the same way so their tests have been combined into NEW

int check_int128(void) {
  return sizeof(__int128); // O32: :[[@LINE]]:17: error: __int128 is not supported on this target
// NEW: ret i32 16
}
