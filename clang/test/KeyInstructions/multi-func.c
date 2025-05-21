// RUN: %clang_cc1 -gkey-instructions -gno-column-info -x c++ %s -debug-info-kind=line-tables-only -emit-llvm -o - \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

// RUN: %clang_cc1 -gkey-instructions -gno-column-info -x c %s -debug-info-kind=line-tables-only -emit-llvm -o - \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

// Check atomGroup is reset to start at 1 in each function.

// CHECK: ret{{.*}}, !dbg [[AG:!.*]]
void a() {}

// CHECK: ret{{.*}}, !dbg [[BG:!.*]]
void b() {}

// CHECK: [[A:!.*]] = distinct !DISubprogram(name: "a",
// CHECK: [[AG]] = !DILocation({{.*}}, scope: [[A]], atomGroup: 1, atomRank: 1)
// CHECK: [[B:!.*]] = distinct !DISubprogram(name: "b",
// CHECK: [[BG]] = !DILocation({{.*}}, scope: [[B]], atomGroup: 1, atomRank: 1)
