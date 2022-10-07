// RUN: %clang_pgogen -o %t -g -mllvm --debug-info-correlate -mllvm --disable-vp=true %s
// RUN: llvm-profdata show --debug-info=%t --detailed-summary --show-prof-sym-list | FileCheck %s
// RUN: llvm-profdata show --debug-info=%t --output-format=yaml | FileCheck %s --match-full-lines --check-prefix YAML

// RUN: %clang_pgogen -o %t.no.dbg -mllvm --debug-info-correlate -mllvm --disable-vp=true %s
// RUN: not llvm-profdata show --debug-info=%t.no.dbg 2>&1 | FileCheck %s --check-prefix NO-DBG
// NO-DBG: unable to correlate profile: could not find any profile metadata in debug info

// CHECK: a
// YAML: Probes:
// YAML:   - Function Name:   a
// YAML:     Linkage Name:    a
// YAML:     CFG Hash:        0x[[#%.1X,HASH:]]
// YAML:     Counter Offset:  0x0
// YAML:     Num Counters:    1
// YAML:     File:            [[FILE:'.*']]
// YAML:     Line:            [[@LINE+1]]
void a() {}

// CHECK: b
// YAML:   - Function Name:   b
// YAML:     Linkage Name:    b
// YAML:     CFG Hash:        0x[[#%.1X,HASH:]]
// YAML:     Counter Offset:  0x8
// YAML:     Num Counters:    1
// YAML:     File:            [[FILE:'.*']]
// YAML:     Line:            [[@LINE+1]]
void b() {}

// CHECK: main
// YAML:   - Function Name:   main
// YAML:     Linkage Name:    main
// YAML:     CFG Hash:        0x[[#%.1X,HASH:]]
// YAML:     Counter Offset:  0x10
// YAML:     Num Counters:    1
// YAML:     File:            [[FILE]]
// YAML:     Line:            [[@LINE+1]]
int main() { return 0; }

// CHECK: Counters section size: 0x18 bytes
// CHECK: Found 3 functions
