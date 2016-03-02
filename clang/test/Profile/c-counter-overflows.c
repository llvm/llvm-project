// Test that big branch weights get scaled down to 32-bits, rather than just
// truncated.

// RUN: llvm-profdata merge %S/Inputs/c-counter-overflows.proftext -o %t.profdata
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.9 -main-file-name c-counter-overflows.c %s -o - -emit-llvm -fprofile-instrument-use-path=%t.profdata | FileCheck %s

typedef unsigned long long uint64_t;

int main(int argc, const char *argv[]) {
  // Need counts higher than 32-bits.
  // CHECK: br {{.*}} !prof ![[FOR:[0-9]+]]
  // max   = 0xffffffff0
  // scale = 0xffffffff0 / 0xffffffff + 1 = 17
  // loop-body: 0xffffffff0 / 17 + 1 = 0xf0f0f0f0 + 1 = 4042322161 => -252645135
  // loop-exit: 0x000000001 / 17 + 1 = 0x00000000 + 1 =          1 =>          1
  for (uint64_t I = 0; I < 0xffffffff0; ++I) {
    // max   = 0xffffffff * 15 = 0xefffffff1
    // scale = 0xefffffff1 / 0xffffffff + 1 = 16
    // CHECK: br {{.*}} !prof ![[IF:[0-9]+]]
    if (I & 0xf) {
      // 0xefffffff1 / 16 + 1 = 0xefffffff + 1 = 4026531840 => -268435456
    } else {
      // 0x0ffffffff / 16 + 1 = 0x0fffffff + 1 =  268435456 =>  268435456
    }

    // max   = 0xffffffff * 5 = 0x4fffffffb
    // scale = 0x4fffffffb / 0xffffffff + 1 = 6
    // CHECK: ], !prof ![[SWITCH:[0-9]+]]
    switch ((I & 0xf) / 5) {
    case 0:
      // 0x4fffffffb / 6 = 0xd5555554 + 1 = 3579139413 => -715827883
      break;
    case 1:
      // 0x4fffffffb / 6 = 0xd5555554 + 1 = 3579139413 => -715827883
      break;
    case 2:
      // 0x4fffffffb / 6 = 0xd5555554 + 1 = 3579139413 => -715827883
      break;
    default:
      // 0x0ffffffff / 6 = 0x2aaaaaaa + 1 =  715827883 =>  715827883
      break;
    }
  }
  return 0;
}

// CHECK-DAG: ![[FOR]] = !{!"branch_weights", i32 -252645135, i32 1}
// CHECK-DAG: ![[IF]]  = !{!"branch_weights", i32 -268435456, i32 268435456}
// CHECK-DAG: ![[SWITCH]] = !{!"branch_weights", i32 715827883, i32 -715827883, i32 -715827883, i32 -715827883}
