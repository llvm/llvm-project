// RUN: %clang_cc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck %s

struct Tuple {
  int Fld_1;
  int Fld_2;
};
__attribute__((optnone)) Tuple get() { return {10, 20}; }

// CHECK-LABEL: define dso_local noundef i32 @main
// CHECK:      %retval = alloca i32, align 4
// CHECK-NEXT: [[T0:%.*]] = alloca %struct.Tuple, align 4
// CHECK:      call void @llvm.dbg.declare(metadata ptr [[T0]], metadata {{.*}}, metadata !DIExpression())
// CHECK:      call void @llvm.dbg.declare(metadata ptr [[T0]], metadata {{.*}}, metadata !DIExpression(DW_OP_plus_uconst, {{[0-9]+}}))
// CHECK-NOT:  call void @llvm.dbg.declare(metadata ptr [[T0]], metadata {{.*}}, metadata !DIExpression())
//
int main() {
  auto [Var_1, Var_2] = get();

  return Var_1 + Var_2;
}
