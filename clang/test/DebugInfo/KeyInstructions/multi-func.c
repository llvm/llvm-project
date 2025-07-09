// RUN: %clang_cc1 -triple x86_64-linux-gnu -gkey-instructions -gno-column-info -x c++ %s -debug-info-kind=line-tables-only -emit-llvm -o - \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

// RUN: %clang_cc1 -triple x86_64-linux-gnu -gkey-instructions -gno-column-info -x c %s -debug-info-kind=line-tables-only -emit-llvm -o - \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

// Check atomGroup is reset to start at 1 in each function.

int g;
// CHECK: store{{.*}}, !dbg [[AG:!.*]]
// CHECK: ret{{.*}}, !dbg [[ARET:!.*]]
void a() { g = 0; }

// CHECK: store{{.*}}, !dbg [[BG:!.*]]
// CHECK: ret{{.*}}, !dbg [[BRET:!.*]]
void b() { g = 0; }

// CHECK: [[A:!.*]] = distinct !DISubprogram(name: "a",
// CHECK: [[AG]] = !DILocation(line: 12, scope: [[A]], atomGroup: 1, atomRank: 1)
// CHECK: [[ARET]] = !DILocation(line: 12, scope: [[A]], atomGroup: 2, atomRank: 1)

// CHECK: [[B:!.*]] = distinct !DISubprogram(name: "b",
// CHECK: [[BG]] = !DILocation(line: 16, scope: [[B]], atomGroup: 1, atomRank: 1)
// CHECK: [[BRET]] = !DILocation(line: 16, scope: [[B]], atomGroup: 2, atomRank: 1)
