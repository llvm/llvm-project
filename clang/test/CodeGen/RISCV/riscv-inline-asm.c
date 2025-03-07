// RUN: %clang_cc1 -triple riscv32 -O2 -emit-llvm %s -o - \
// RUN:     | FileCheck %s
// RUN: %clang_cc1 -triple riscv64 -O2 -emit-llvm %s -o - \
// RUN:     | FileCheck %s

// Test RISC-V specific inline assembly constraints and modifiers.

long test_r(long x) {
// CHECK-LABEL: define{{.*}} {{i64|i32}} @test_r(
// CHECK: call {{i64|i32}} asm sideeffect "", "=r,r"({{i64|i32}} %{{.*}})
  long ret;
  asm volatile ("" : "=r"(ret) : "r"(x));
// CHECK: call {{i64|i32}} asm sideeffect "", "=r,r"({{i64|i32}} %{{.*}})
  asm volatile ("" : "=r"(ret) : "r"(x));
  return ret;
}

long test_cr(long x) {
// CHECK-LABEL: define{{.*}} {{i64|i32}} @test_cr(
// CHECK: call {{i64|i32}} asm sideeffect "", "=^cr,^cr"({{i64|i32}} %{{.*}})
  long ret;
  asm volatile ("" : "=cr"(ret) : "cr"(x));
  return ret;
}

float cf;
double cd;
void test_cf(float f, double d) {
// CHECK-LABEL: define{{.*}} void @test_cf(
// CHECK: call float asm sideeffect "", "=^cf,^cf"(float %{{.*}})
  asm volatile("" : "=cf"(cf) : "cf"(f));
// CHECK: call double asm sideeffect "", "=^cf,^cf"(double %{{.*}})
  asm volatile("" : "=cf"(cd) : "cf"(d));
}

#if __riscv_xlen == 32
typedef long long double_xlen_t;
#elif __riscv_xlen == 64
typedef __int128_t double_xlen_t;
#endif
double_xlen_t test_R_wide_scalar(double_xlen_t p) {
// CHECK-LABEL: define{{.*}} {{i128|i64}} @test_R_wide_scalar(
// CHECK: call {{i128|i64}} asm sideeffect "", "=R,R"({{i128|i64}} %{{.*}})
  double_xlen_t ret;
  asm volatile("" : "=R"(ret) : "R"(p));
  return ret;
}

double_xlen_t test_cR_wide_scalar(double_xlen_t p) {
// CHECK-LABEL: define{{.*}} {{i128|i64}} @test_cR_wide_scalar(
// CHECK: call {{i128|i64}} asm sideeffect "", "=^cR,^cR"({{i128|i64}} %{{.*}})
  double_xlen_t ret;
  asm volatile("" : "=cR"(ret) : "cR"(p));
  return ret;
}

void test_I(void) {
// CHECK-LABEL: define{{.*}} void @test_I()
// CHECK: call void asm sideeffect "", "I"(i32 2047)
  asm volatile ("" :: "I"(2047));
// CHECK: call void asm sideeffect "", "I"(i32 -2048)
  asm volatile ("" :: "I"(-2048));
}

void test_J(void) {
// CHECK-LABEL: define{{.*}} void @test_J()
// CHECK: call void asm sideeffect "", "J"(i32 0)
  asm volatile ("" :: "J"(0));
}

void test_K(void) {
// CHECK-LABEL: define{{.*}} void @test_K()
// CHECK: call void asm sideeffect "", "K"(i32 31)
  asm volatile ("" :: "K"(31));
// CHECK: call void asm sideeffect "", "K"(i32 0)
  asm volatile ("" :: "K"(0));
}

float f;
double d;
void test_f(void) {
// CHECK-LABEL: define{{.*}} void @test_f()
// CHECK: [[FLT_ARG:%[a-zA-Z_0-9]+]] = load float, ptr @f
// CHECK: call void asm sideeffect "", "f"(float [[FLT_ARG]])
  asm volatile ("" :: "f"(f));
// CHECK: [[FLT_ARG:%[a-zA-Z_0-9]+]] = load double, ptr @d
// CHECK: call void asm sideeffect "", "f"(double [[FLT_ARG]])
  asm volatile ("" :: "f"(d));
}

void test_A(int *p) {
// CHECK-LABEL: define{{.*}} void @test_A(ptr noundef %p)
// CHECK: call void asm sideeffect "", "*A"(ptr elementtype(i32) %p)
  asm volatile("" :: "A"(*p));
}

extern int var, arr[2][2];
struct Pair { int a, b; } pair;

// CHECK-LABEL: test_s(
// CHECK:         call void asm sideeffect "// $0 $1 $2", "s,s,s"(ptr nonnull @var, ptr nonnull getelementptr inbounds nuw (i8, ptr @arr, {{.*}}), ptr nonnull @test_s)
// CHECK:         call void asm sideeffect "// $0", "s"(ptr nonnull getelementptr inbounds nuw (i8, ptr @pair, {{.*}}))
// CHECK:         call void asm sideeffect "// $0 $1 $2", "S,S,S"(ptr nonnull @var, ptr nonnull getelementptr inbounds nuw (i8, ptr @arr, {{.*}}), ptr nonnull @test_s)
void test_s(void) {
  asm("// %0 %1 %2" :: "s"(&var), "s"(&arr[1][1]), "s"(test_s));
  asm("// %0" :: "s"(&pair.b));

  asm("// %0 %1 %2" :: "S"(&var), "S"(&arr[1][1]), "S"(test_s));
}

// CHECK-LABEL: test_modifiers(
// CHECK: call void asm sideeffect "// ${0:i} ${1:i}", "r,r"({{i32|i64}} %val, i32 37)
// CHECK: call void asm sideeffect "// ${0:z} ${1:z}", "i,i"(i32 0, i32 1)
// CHECK: call void asm sideeffect "// ${0:N}", "r"({{i32|i64}} %val)
void test_modifiers(long val) {
  asm volatile("// %i0 %i1" :: "r"(val), "r"(37));
  asm volatile("// %z0 %z1" :: "i"(0), "i"(1));
  asm volatile("// %N0" :: "r"(val));
}
