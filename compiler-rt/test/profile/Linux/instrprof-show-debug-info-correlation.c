// RUN: %clang_pgogen -o %t -g -mllvm --debug-info-correlate -mllvm --disable-vp=true %s
// RUN: llvm-profdata show --debug-info=%t --detailed-summary --show-prof-sym-list | FileCheck %s

// RUN: %clang_pgogen -o %t.no.dbg -mllvm --debug-info-correlate -mllvm --disable-vp=true %s
// RUN: not llvm-profdata show --debug-info=%t.no.dbg 2>&1 | FileCheck %s --check-prefix NO-DBG
// NO-DBG: unable to correlate profile: could not find any profile metadata in debug info

void a() {}
void b() {}
int main() { return 0; }

// CHECK: a
// CHECK: b
// CHECK: main
// CHECK: Counters section size: 0x18 bytes
// CHECK: Found 3 functions
