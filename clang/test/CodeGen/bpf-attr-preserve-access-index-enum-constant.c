// REQUIRES: bpf-registered-target
// RUN: %clang_cc1 -triple bpf -debug-info-kind=limited -emit-llvm -disable-llvm-passes \
// RUN:   -Wno-bpf-enum-not-relocated %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple bpf -emit-llvm -disable-llvm-passes \
// RUN:   -Wno-bpf-enum-not-relocated %s -o - \
// RUN:   | FileCheck --check-prefix=NODBG --implicit-check-not=@llvm.bpf.preserve.enum.value %s

// With -Wno-bpf-enum-not-relocated, an enumerator of an enum with the
// preserve_access_index attribute is a constant where a constant is required,
// and its value in the local enum is used. Elsewhere, it is still relocated,
// unless there is no -g to relocate against (NODBG).
enum e1 { E1_A = 1, E1_B = 2 } __attribute__((preserve_access_index));

struct ops { unsigned long flags; };

// CHECK-DAG: @arr = global [2 x i32] zeroinitializer
int arr[E1_A + 1];
// CHECK-DAG: @ops = global %struct.ops { i64 2 }
struct ops ops = { .flags = E1_B };
// CHECK-DAG: @locals.s = internal global i32 2
// CHECK-DAG: @__const.locals.a = private unnamed_addr constant [4 x i32] [i32 0, i32 0, i32 1, i32 0]
// CHECK-DAG: @[[E1_A:[0-9]+]] = private unnamed_addr constant [7 x i8] c"E1_A:1\00"

int test(int x) {
  asm volatile("r0 = %0" :: "i"(E1_B) : "r0");
  return x == E1_B;
}

// CHECK-LABEL: define dso_local i32 @test(
// CHECK: call void asm sideeffect "r0 = $0", "i,~{r0}"(i32 2)
// CHECK: call i64 @llvm.bpf.preserve.enum.value(

// NODBG-LABEL: define dso_local i32 @test(
// NODBG: icmp eq i32 %{{[0-9a-z]+}}, 2

int locals(void) {
  static int s = E1_B;
  int a[4] = {[E1_B] = 1};
  int l = E1_A;
  return s + a[0] + l;
}

// CHECK-LABEL: define dso_local i32 @locals(
// CHECK: call void @llvm.memcpy.{{.*}}(ptr {{.*}}, ptr {{.*}}@__const.locals.a
// CHECK: call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @[[E1_A]], i64 1)
// CHECK-NOT: call i64 @llvm.bpf.preserve.enum.value(
// CHECK: ret i32

// NODBG-LABEL: define dso_local i32 @locals(
// NODBG: store i32 1, ptr %l,
