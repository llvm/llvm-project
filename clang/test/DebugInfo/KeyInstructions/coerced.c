// RUN: %clang -gkey-instructions -gno-column-info -x c++ %s -gmlt -S -emit-llvm -o - -target x86_64-unknown-linux \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

// RUN: %clang -gkey-instructions -gno-column-info -x c %s -gmlt -S -emit-llvm -o - -target x86_64-unknown-linux \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

typedef struct {
  void* a;
  void* b;
} Struct;

Struct get();
void store() {
  // CHECK: %1 = extractvalue { ptr, ptr } %call, 0, !dbg [[G1R2:!.*]]
  // CHECK: store ptr %1, ptr {{.*}}, !dbg [[G1R1:!.*]]
  // CHECK: %3 = extractvalue { ptr, ptr } %call, 1, !dbg [[G1R2]]
  // CHECK: store ptr %3, ptr {{.*}}, !dbg [[G1R1:!.*]]
  Struct s = get();
  // CHECK: ret void, !dbg [[G2R1:!.*]]
}

// CHECK: [[G1R2]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 2)
// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[G2R1]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
