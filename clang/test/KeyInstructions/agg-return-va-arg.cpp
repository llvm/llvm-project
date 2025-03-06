// RUN: %clang %s -gmlt -gcolumn-info -S -emit-llvm -o - -Wno-unused-variable \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

typedef struct {
  struct{} a;
  double b;
} s1;

s1 f(int z, ...) {
  __builtin_va_list list;
  __builtin_va_start(list, z);
// CHECK: vaarg.end:
// CHECK-NEXT: %vaarg.addr = phi ptr
// CHECK-NEXT: call void @llvm.memcpy{{.*}}, !dbg [[COL10_G1R2:!.*]]
// CHECK-NEXT: %3 = getelementptr{{.*}}
// CHECK-NEXT: %4 = load double, ptr %3{{.*}} !dbg [[COL3_G1R2:!.*]]
// CHECK-NEXT: ret double %4, !dbg [[G1R1:!.*]]
  return __builtin_va_arg(list, s1);
}

// CHECK: [[COL10_G1R2]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 2)
// CHECK: [[COL3_G1R2]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 2)
// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
