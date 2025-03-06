// RUN: %clang %s -gmlt -gcolumn-info -S -emit-llvm -o - -Wno-unused-variable \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

struct S {
  void* a;
  void* b;
};
S get();
void fun() {
// CHECK: %0 = getelementptr inbounds { ptr, ptr }, ptr %s, i32 0, i32 0
// CHECK: %1 = extractvalue { ptr, ptr } %call, 0, !dbg [[G1R2:!.*]]
// CHECK: store ptr %1, ptr %0{{.*}}, !dbg [[G1R1:!.*]]
// CHECK: %2 = getelementptr inbounds { ptr, ptr }, ptr %s, i32 0, i32 1
// CHECK: %3 = extractvalue { ptr, ptr } %call, 1, !dbg [[G1R2]]
// CHECK: store ptr %3, ptr %2{{.*}}, !dbg [[G1R1:!.*]]
  S s = get();
// CHECK: ret void, !dbg [[G2R1:!.*]]
}

// CHECK: [[G1R2]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 2)
// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[G2R1]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)

