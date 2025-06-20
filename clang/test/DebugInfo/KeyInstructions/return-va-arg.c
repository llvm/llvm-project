// RUN: %clang_cc1 -triple x86_64-linux-gnu -gkey-instructions -gno-column-info -x c++ %s -debug-info-kind=line-tables-only -emit-llvm -o - \
// RUN: | FileCheck %s --implicit-check-not atomGroup --implicit-check-not atomRank

// RUN: %clang_cc1 -triple x86_64-linux-gnu -gkey-instructions -gno-column-info -x c %s -debug-info-kind=line-tables-only -emit-llvm -o - \
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
// CHECK-NEXT: call void @llvm.memcpy{{.*}}, !dbg [[G1R1:!.*]]
// CHECK-NEXT: {{.*}} = getelementptr{{.*}}
// CHECK-NEXT: [[LOAD:%.*]] = load double{{.*}}, !dbg [[G1R2:!.*]]
// CHECK-NEXT: ret double [[LOAD]], !dbg [[G1R1]]
  return __builtin_va_arg(list, s1);
}

// CHECK: [[G1R1]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 1)
// CHECK: [[G1R2]] = !DILocation({{.*}}, atomGroup: 1, atomRank: 2)
