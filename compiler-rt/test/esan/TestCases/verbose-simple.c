// RUN: %clang_esan_frag -O0 %s -o %t 2>&1
// RUN: %env_esan_opts="verbosity=1 log_exe_name=1" %run %t 2>&1 | FileCheck %s

int main(int argc, char **argv) {
  // CHECK:      in esan::initializeLibrary
  // (There can be a re-exec for stack limit here.)
  // CHECK:      Shadow scale=2 offset=0x440000000000
  // CHECK-NEXT: Shadow #0: [110000000000-114000000000) (256GB)
  // CHECK-NEXT: Shadow #1: [124000000000-12c000000000) (512GB)
  // CHECK-NEXT: Shadow #2: [148000000000-150000000000) (512GB)
  // CHECK-NEXT: in esan::finalizeLibrary
  // CHECK-NEXT: ==verbose-simple{{.*}}EfficiencySanitizer: total struct field access count = 0
  return 0;
}
