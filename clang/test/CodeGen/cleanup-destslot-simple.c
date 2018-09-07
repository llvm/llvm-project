// RUN: %clang_cc1 -O1 -triple x86_64-none-linux-gnu -emit-llvm -debug-info-kind=line-tables-only %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=LIFETIME

// We shouldn't have markers at -O0 or with msan.
// RUN: %clang_cc1 -O0 -triple x86_64-none-linux-gnu -emit-llvm -debug-info-kind=line-tables-only %s -o - | FileCheck %s
// RUN: %clang_cc1 -O1 -triple x86_64-none-linux-gnu -emit-llvm -debug-info-kind=line-tables-only %s -o - -fsanitize=memory | FileCheck %s
// RUN: %clang_cc1 -O1 -triple x86_64-none-linux-gnu -emit-llvm -debug-info-kind=line-tables-only %s -o - -fsanitize=kernel-memory | FileCheck %s

// There is no exception to handle here, lifetime.end is not a destructor,
// so there is no need have cleanup dest slot related code
// CHECK-LABEL: define i32 @test
int test() {
  int x = 3;
  int *volatile p = &x;
  return *p;
// CHECK: [[X:%.*]] = alloca i32
// CHECK: [[P:%.*]] = alloca i32*
// LIFETIME: call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %{{.*}}){{( #[0-9]+)?}}, !dbg
// LIFETIME: call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %{{.*}}){{( #[0-9]+)?}}, !dbg
// CHECK-NOT: store i32 %{{.*}}, i32* %cleanup.dest.slot
// LIFETIME: call void @llvm.lifetime.end.p0i8(i64 8, {{.*}}){{( #[0-9]+)?}}, !dbg
// LIFETIME: call void @llvm.lifetime.end.p0i8(i64 4, {{.*}}){{( #[0-9]+)?}}, !dbg
}
