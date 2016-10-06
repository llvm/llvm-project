// RUN: %clang_esan_frag -O0 %s -o %t 2>&1
// RUN: %env_esan_opts="verbosity=1 log_exe_name=1" %run %t 2>&1 | FileCheck --check-prefix=%arch --check-prefix=CHECK %s

int main(int argc, char **argv) {
  // CHECK:      in esan::initializeLibrary
  // (There can be a re-exec for stack limit here.)
  // x86_64:      Shadow scale=2 offset=0x440000000000
  // x86_64-NEXT: Shadow #0: [110000000000-114000000000) (256GB)
  // x86_64-NEXT: Shadow #1: [124000000000-12c000000000) (512GB)
  // x86_64-NEXT: Shadow #2: [148000000000-150000000000) (512GB)
  // mips64:      Shadow scale=2 offset=0x4400000000
  // mips64-NEXT: Shadow #0: [1140000000-1180000000) (1GB)
  // mips64-NEXT: Shadow #1: [1380000000-13c0000000) (1GB)
  // mips64-NEXT: Shadow #2: [14c0000000-1500000000) (1GB)
  // CHECK: in esan::finalizeLibrary
  // CHECK: ==verbose-simple{{.*}}EfficiencySanitizer: total struct field access count = 0
  return 0;
}
