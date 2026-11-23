// RUN: %clang_cc1 -gkey-instructions -gno-column-info -x c++ %s -debug-info-kind=line-tables-only -emit-llvm -o - -triple x86_64-unknown-linux \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank --check-prefixes=CHECK,CHECK-CXX

// RUN: %clang_cc1 -gkey-instructions -gno-column-info -x c %s -debug-info-kind=line-tables-only -emit-llvm -o - -triple x86_64-unknown-linux \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank --check-prefixes=CHECK,CHECK-C

typedef struct {
  void* a;
  void* b;
} Struct;
Struct get();

void test() {
// CHECK: %1 = extractvalue { ptr, ptr } %call, 0, !dbg [[G1R2:!.*]]
// CHECK: store ptr %1, ptr {{.*}}, !dbg [[G1R1:!.*]]
// CHECK: %3 = extractvalue { ptr, ptr } %call, 1, !dbg [[G1R2]]
// CHECK: store ptr %3, ptr {{.*}}, !dbg [[G1R1:!.*]]
  Struct s = get();
// CHECK: ret{{.*}}, !dbg [[RET:!.*]]
}

typedef struct { int i; } Int;
Int getInt(void);

// CHECK-C: @test2
// CHECK-CXX: @_Z5test2v
void test2() {
// CHECK: %call = call i32 @{{(_Z6)?}}getInt{{v?}}(), !dbg [[T2_G1R2:!.*]]
// CHECK: [[gep:%.*]] = getelementptr inbounds nuw %struct.Int, ptr %i, i32 0, i32 0
// CHECK: store i32 %call, ptr [[gep]]{{.*}}, !dbg [[T2_G1R1:!.*]]
// CHECK: ret{{.*}}, !dbg [[T2_RET:!.*]]
  Int i = getInt();
}

// CHECK: [[G1R2]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 2)
// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[RET]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
// CHECK: [[T2_G1R2]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 2)
// CHECK: [[T2_G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[T2_RET]] = !DILocation({{.*}}, atomGroup: 2, atomRank: 1)
