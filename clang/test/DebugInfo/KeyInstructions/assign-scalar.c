// RUN: %clang_cc1 -gkey-instructions -x c++ %s -debug-info-kind=line-tables-only -emit-llvm -o - \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

// RUN: %clang_cc1 -gkey-instructions -x c %s -debug-info-kind=line-tables-only -emit-llvm -o - \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

unsigned long long g, h, i;
void fun() {
// CHECK: store i64 0, ptr @g{{.*}}, !dbg [[G1R1:!.*]]
    g = 0;

// Treat the two assignments as two atoms.
//
// FIXME: Because of the atomGroup implementation the load can only be
// associated with one of the two stores, despite being a good backup
// loction for both.
// CHECK-NEXT: %0 = load i64, ptr @g{{.*}}, !dbg [[G2R2:!.*]]
// CHECK-NEXT: store i64 %0, ptr @g{{.*}}, !dbg [[G3R1:!.*]]
// CHECK-NEXT: store i64 %0, ptr @g{{.*}}, !dbg [[G2R1:!.*]]
    g = g = g;

// Compound assignment.
// CHECK: %1 = load i64, ptr @g
// CHECK: %add = add i64 %1, 50, !dbg [[G4R2:!.*]]
// CHECK: store i64 %add, ptr @g{{.*}}, !dbg [[G4R1:!.*]]
    g += 50;

// Pre/Post Inc/Dec.
// CHECK: %2 = load i64, ptr @g
// CHECK: %inc = add i64 %2, 1, !dbg [[G5R2:!.*]]
// CHECK: store i64 %inc, ptr @g{{.*}}, !dbg [[G5R1:!.*]]
    ++g;
// CHECK: %3 = load i64, ptr @g
// CHECK: %dec = add i64 %3, -1, !dbg [[G6R2:!.*]]
// CHECK: store i64 %dec, ptr @g{{.*}}, !dbg [[G6R1:!.*]]
    g--;

// Compound assignment with assignment on RHS, the assignments should have
// their own separate atom groups.
// CHECK-NEXT: %4 = load i64, ptr @h{{.*}}, !dbg [[load_h_loc:!.*]]
// CHECK-NEXT: %inc1 = add i64 %4, 1, !dbg [[G8R2:!.*]]
// CHECK-NEXT: store i64 %inc1, ptr @h{{.*}}, !dbg [[G8R1:!.*]]
// CHECK-NEXT: %5 = load i64, ptr @g{{.*}}, !dbg [[load_g_loc:!.*]]
// CHECK-NEXT: %add2 = add i64 %5, %inc1, !dbg [[G7R2:!.*]]
// CHECK-NEXT: store i64 %add2, ptr @g{{.*}}, !dbg [[G7R1:!.*]]
    g += ++h;

// Double check the comma operator doesn't disturb atom groupings. There
// are three assignments, so we should get three groups.
// FIXME: Same situation as earlier in the test - because of the atomGroup
// implementation the load (from h) can only be associated with one of the two
// stores (to h and g) despite being a good backup location for both.
// CHECK-NEXT: %6 = load i64, ptr @h{{.*}}, !dbg [[load_h_loc2:!.*]]
// CHECK-NEXT: %inc3 = add i64 %6, 1, !dbg [[G9R2:!.*]]
// CHECK-NEXT: store i64 %inc3, ptr @h{{.*}}, !dbg [[G10R1:!.*]]
// CHECK-NEXT: store i64 %inc3, ptr @g{{.*}}, !dbg [[G9R1:!.*]]
// CHECK-NEXT: %7 = load i64, ptr @i{{.*}}, !dbg [[load_i_loc:!.*]]
// CHECK-NEXT: %inc4 = add i64 %7, 1, !dbg [[G11R2:!.*]]
// CHECK-NEXT: store i64 %inc4, ptr @i{{.*}}, !dbg [[G11R1:!.*]]
    g = ++h, ++i;
}

// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[G2R2]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 2)
// CHECK: [[G3R1]] = !DILocation({{.*}}, atomGroup: 3, atomRank: 1)
// CHECK: [[G2R1]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
// CHECK: [[G4R2]] = !DILocation({{.*}}, atomGroup: 4, atomRank: 2)
// CHECK: [[G4R1]] = !DILocation({{.*}}, atomGroup: 4, atomRank: 1)
// CHECK: [[G5R2]] = !DILocation({{.*}}, atomGroup: 5, atomRank: 2)
// CHECK: [[G5R1]] = !DILocation({{.*}}, atomGroup: 5, atomRank: 1)
// CHECK: [[G6R2]] = !DILocation({{.*}}, atomGroup: 6, atomRank: 2)
// CHECK: [[G6R1]] = !DILocation({{.*}}, atomGroup: 6, atomRank: 1)
// CHECK: [[load_h_loc]] = !DILocation(line: [[#]], column: [[#]], scope: ![[#]])
// CHECK: [[G8R2]] = !DILocation({{.*}}, atomGroup: 8, atomRank: 2)
// CHECK: [[G8R1]] = !DILocation({{.*}}, atomGroup: 8, atomRank: 1)
// CHECK: [[load_g_loc]] = !DILocation(line: [[#]], column: [[#]], scope: ![[#]])
// CHECK: [[G7R2]] = !DILocation({{.*}}, atomGroup: 7, atomRank: 2)
// CHECK: [[G7R1]] = !DILocation({{.*}}, atomGroup: 7, atomRank: 1)
// CHECK: [[load_h_loc2]] = !DILocation(line: [[#]], column: [[#]], scope: ![[#]])
// CHECK: [[G9R2]] = !DILocation({{.*}}, atomGroup: 9, atomRank: 2)
// CHECK: [[G10R1]] = !DILocation({{.*}}, atomGroup: 10, atomRank: 1)
// CHECK: [[G9R1]] = !DILocation({{.*}}, atomGroup: 9, atomRank: 1)
// CHECK: [[load_i_loc]] = !DILocation(line: [[#]], column: [[#]], scope: ![[#]])
// CHECK: [[G11R2]] = !DILocation({{.*}}, atomGroup: 11, atomRank: 2)
// CHECK: [[G11R1]] = !DILocation({{.*}}, atomGroup: 11, atomRank: 1)
