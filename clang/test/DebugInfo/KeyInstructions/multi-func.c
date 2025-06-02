// RUN: %clang_cc1 -triple x86_64-linux-gnu -gkey-instructions -gno-column-info -x c++ %s -debug-info-kind=line-tables-only -emit-llvm -o - \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

// RUN: %clang_cc1 -triple x86_64-linux-gnu -gkey-instructions -gno-column-info -x c %s -debug-info-kind=line-tables-only -emit-llvm -o - \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

// Check atomGroup is reset to start at 1 in each function.

int g;
// CHECK: store{{.*}}, !dbg [[AG:!.*]]
void a() { g = 0; }

// CHECK: store{{.*}}, !dbg [[BG:!.*]]
void b() { g = 0; }

// CHECK: [[A:!.*]] = distinct !DISubprogram(name: "a",
// CHECK: [[AG]] = !DILocation(line: 11, scope: [[A]], atomGroup: 1, atomRank: 1)
// CHECK: [[B:!.*]] = distinct !DISubprogram(name: "b",
// CHECK: [[BG]] = !DILocation(line: 14, scope: [[B]], atomGroup: 1, atomRank: 1)
