// Test that we get the expected module flag metadata for the memory profile
// filename.
// RUN: %clang -target x86_64-linux-gnu -S -emit-llvm -o - %s | FileCheck %s --check-prefix=NONE
// RUN: %clang -target x86_64-linux-gnu -fmemory-profile -S -emit-llvm -o - %s | FileCheck %s --check-prefix=DEFAULTNAME
// RUN: %clang -target x86_64-linux-gnu -fmemory-profile=/tmp -S -emit-llvm -o - %s | FileCheck %s --check-prefix=CUSTOMNAME
int main(void) {
  return 0;
}

// NONE-NOT: MemProfProfileFilename
// DEFAULTNAME: !{i32 1, !"MemProfProfileFilename", !"memprof.profraw"}
// CUSTOMNAME: !{i32 1, !"MemProfProfileFilename", !"/tmp{{.*}}memprof.profraw"}
