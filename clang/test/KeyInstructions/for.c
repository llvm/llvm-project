// RUN: %clang -gkey-instructions -x c++ %s -gmlt -S -emit-llvm -o - \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

// RUN: %clang -gkey-instructions -x c %s -gmlt -S -emit-llvm -o -  \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

// Perennial quesiton: should the inc be its own source atom or not
// (currently it is).

// FIXME: See do.c and while.c regarding cmp and cond br groups.

void a(int A) {
// CHECK: entry:
// CHECK: store i32 0, ptr %i{{.*}}, !dbg [[G1R1:!.*]]
// CHECK: for.cond:
// CHECK: %cmp = icmp slt i32 %0, %1, !dbg [[G2R1:!.*]]
// CHECK: br i1 %cmp, label %for.body, label %for.end, !dbg [[G3R1:!.*]]

// FIXME: Added uncond br group here which is useful for O0, which we're
// no longer targeting. With optimisations loop rotate puts the condition
// into for.inc and simplifycfg smooshes that and for.body together, so
// it's not clear whether it adds any value.
// CHECK: for.body:
// CHECK: br label %for.inc, !dbg [[G5R1:!.*]]

// CHECK: for.inc:
// CHECK: %inc = add{{.*}}, !dbg [[G4R2:!.*]]
// CHECK: store i32 %inc, ptr %i{{.*}}, !dbg [[G4R1:!.*]]
    for (int i = 0; i < A; ++i) { }

// CHECK: ret{{.*}}, !dbg [[RET:!.*]]
}

// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[G2R1]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
// CHECK: [[G3R1]] = !DILocation({{.*}}, atomGroup: 3, atomRank: 1)
// CHECK: [[G5R1]] = !DILocation({{.*}}, atomGroup: 5, atomRank: 1)
// CHECK: [[G4R2]] = !DILocation({{.*}}, atomGroup: 4, atomRank: 2)
// CHECK: [[G4R1]] = !DILocation({{.*}}, atomGroup: 4, atomRank: 1)
// CHECK: [[RET:!.*]] = !DILocation({{.*}}, atomGroup: [[#]], atomRank: [[#]])
