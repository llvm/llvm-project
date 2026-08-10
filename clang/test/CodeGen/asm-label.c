// RUN: %clang_cc1 -triple=i686-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,LINUX
// RUN: %clang_cc1 -triple=i686-apple-darwin9 -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,DARWIN

char *strerror(int) asm("alias");
int x __asm("foo");

int *test(void) {
  static int y __asm("bar");
  strerror(-1);
  return &y;
}

// LINUX: @bar = internal global i32 0
// LINUX: @foo ={{.*}} global i32 0
// LINUX: declare ptr @alias(i32 noundef)

// DARWIN: @"\01bar" = internal global i32 0
// DARWIN: @"\01foo" ={{.*}} global i32 0
// DARWIN: declare ptr @"\01alias"(i32 noundef)

extern void *memcpy(void *__restrict, const void *__restrict, unsigned long);
extern __typeof(memcpy) memcpy asm("__GI_memcpy");
void test_memcpy(void *dst, void *src, unsigned long n) {
  memcpy(dst, src, n);
}
// CHECK-LABEL: @test_memcpy(
// LINUX:         call ptr @__GI_memcpy(
// LINUX:       declare ptr @__GI_memcpy(ptr noundef, ptr noundef, i32 noundef)
// DARWIN:        call ptr @"\01__GI_memcpy"(
// DARWIN:      declare ptr @"\01__GI_memcpy"(ptr noundef, ptr noundef, i32 noundef)

long lrint(double x) asm("__GI_lrint");
long test_lrint(double x) {
  return lrint(x);
}
// CHECK-LABEL: @test_lrint(
// LINUX:         call i32 @__GI_lrint(
// LINUX:       declare i32 @__GI_lrint(double noundef)
// DARWIN:        call i32 @"\01__GI_lrint"(
// DARWIN:      declare i32 @"\01__GI_lrint"(double noundef)

/// NOTE: GCC can optimize out abs in -O1 or above. Clang does not
/// communicate the mapping to the backend so the libcall cannot be eliminated.
int abs(int x) asm("__GI_abs");
long test_abs(int x) {
  return abs(x);
}
// CHECK-LABEL: @test_abs(
// LINUX:         call i32 @__GI_abs(

/// FIXME: test_sin should call real_sin instead.
double sin(double x) asm("real_sin");
double test_sin(double d) { return __builtin_sin(d); }
// CHECK-LABEL: @test_sin(
// LINUX:         call double @llvm.sin.f64(
