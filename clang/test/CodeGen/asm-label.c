// RUN: %clang_cc1 -triple=i686-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,LINUX
// RUN: %clang_cc1 -triple=i686-apple-darwin9 -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,DARWIN
// RUN: %clang_cc1 -triple=i686-pc-linux-gnu -fexceptions -emit-llvm %s -o - | FileCheck %s --check-prefix=EH

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

/// A call through the __builtin_ alias of a library function uses the asm
/// label of the library function, as a direct call does.
extern char *stpcpy(char *__restrict, const char *__restrict);
extern __typeof(stpcpy) stpcpy asm("__GI_stpcpy");
char *test_builtin_stpcpy(char *d, const char *s) {
  return __builtin_stpcpy(d, s);
}
// CHECK-LABEL: @test_builtin_stpcpy(
// LINUX:         call ptr @__GI_stpcpy(
// DARWIN:        call ptr @"\01__GI_stpcpy"(

extern void abort(void) __attribute__((noreturn));
extern __typeof(abort) abort asm("__GI_abort");
void test_builtin_abort(int x) {
  if (x)
    __builtin_abort();
}
// CHECK-LABEL: @test_builtin_abort(
// LINUX:         call void @__GI_abort(
// DARWIN:        call void @"\01__GI_abort"(

/// The abort call in the terminate landing pad for an EH cleanup also uses
/// the asm label.
void release(int *);
void use(int *);
void test_cleanup_terminate(void) {
  int v __attribute__((cleanup(release))) = 0;
  use(&v);
}
// EH-LABEL: @test_cleanup_terminate(
// EH:         invoke void @release(
// EH-NEXT:      to label %{{.*}} unwind label %[[TERM:.*]]
// EH:       [[TERM]]:
// EH-NEXT:    landingpad { ptr, i32 }
// EH-NEXT:      catch ptr null
// EH-NEXT:    call void @__GI_abort()
// EH-NOT:   call void @abort()
