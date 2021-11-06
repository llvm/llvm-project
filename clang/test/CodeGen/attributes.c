// RUN: %clang_cc1 -emit-llvm -fcf-protection=branch -triple i386-linux-gnu -o %t %s
// RUN: FileCheck --input-file=%t %s

// CHECK: @t5 = weak{{.*}} global i32 2
int t5 __attribute__((weak)) = 2;

// CHECK: @t13 ={{.*}} global %struct.s0 zeroinitializer, section "SECT"
struct s0 { int x; };
struct s0 t13 __attribute__((section("SECT"))) = { 0 };

// CHECK: @t14.x = internal global i32 0, section "SECT"
void t14(void) {
  static int x __attribute__((section("SECT"))) = 0;
}

// CHECK: @t18 ={{.*}} global i32 1, align 4
extern int t18 __attribute__((weak_import));
int t18 = 1;

// CHECK: @t16 = extern_weak global i32
extern int t16 __attribute__((weak_import));

// CHECK: @t6 = protected global i32 0
int t6 __attribute__((visibility("protected")));

// CHECK: @t12 ={{.*}} global i32 0, section "SECT"
int t12 __attribute__((section("SECT")));

// CHECK: @t9 = weak{{.*}} alias void (...), bitcast (void ()* @__t8 to void (...)*)
void __t8() {}
void t9() __attribute__((weak, alias("__t8")));

// CHECK: declare extern_weak i32 @t15()
int __attribute__((weak_import)) t15(void);
int t17() {
  return t15() + t16;
}

// CHECK: define{{.*}} void @t1() [[NR:#[0-9]+]] {
void t1() __attribute__((noreturn));
void t1() { while (1) {} }

// CHECK: define{{.*}} void @t2() [[NUW:#[0-9]+]] {
void t2() __attribute__((nothrow));
void t2() {}

// CHECK: define weak{{.*}} void @t3() [[NUW]] {
void t3() __attribute__((weak));
void t3() {}

// CHECK: define hidden void @t4() [[NUW]] {
void t4() __attribute__((visibility("hidden")));
void t4() {}

// CHECK: define{{.*}} void @t7() [[NR]] {
void t7() __attribute__((noreturn, nothrow));
void t7() { while (1) {} }

// CHECK: define{{.*}} void @t72() [[COLDDEF:#[0-9]+]] {
void t71(void) __attribute__((cold));
void t72() __attribute__((cold));
void t72() { t71(); }
// CHECK: call void @t71() [[COLDSITE:#[0-9]+]]
// CHECK: declare void @t71() [[COLDDECL:#[0-9]+]]

// CHECK: define{{.*}} void @t82() [[HOTDEF:#[0-9]+]] {
void t81(void) __attribute__((hot));
void t82() __attribute__((hot));
void t82() { t81(); }
// CHECK: call void @t81() [[HOTSITE:#[0-9]+]]
// CHECK: declare void @t81() [[HOTDECL:#[0-9]+]]

// CHECK: define{{.*}} void @t10() [[NUW]] section "xSECT" {
void t10(void) __attribute__((section("xSECT")));
void t10(void) {}
// CHECK: define{{.*}} void @t11() [[NUW]] section "xSECT" {
void __attribute__((section("xSECT"))) t11(void) {}

// CHECK: define{{.*}} i32 @t19() [[NUW]] {
extern int t19(void) __attribute__((weak_import));
int t19(void) {
// RUN: %clang_cc1 -emit-llvm -fcf-protection=branch -triple i386-linux-gnu -o %t %s
// RUN: %clang_cc1 -emit-llvm -fcf-protection=branch -triple i386-linux-gnu -o %t %s
// RUN: %clang_cc1 -emit-llvm -fcf-protection=branch -triple i386-linux-gnu -o %t %s
  return 10;
}

// CHECK:define{{.*}} void @t20() [[NUW]] {
// CHECK: call void @abort()
// CHECK-NEXT: unreachable
void t20(void) {
  __builtin_abort();
}

void (__attribute__((fastcall)) *fptr)(int);
void t21(void) {
  fptr(10);
}
// CHECK: [[FPTRVAR:%[a-z0-9]+]] = load void (i32)*, void (i32)** @fptr
// CHECK-NEXT: call x86_fastcallcc void [[FPTRVAR]](i32 inreg noundef 10)


// PR9356: We might want to err on this, but for now at least make sure we
// use the section in the definition.
void __attribute__((section(".foo"))) t22(void);
void __attribute__((section(".bar"))) t22(void) {}

// CHECK: define{{.*}} void @t22() [[NUW]] section ".bar"

// CHECK: define{{.*}} void @t23() [[NOCF_CHECK_FUNC:#[0-9]+]]
void __attribute__((nocf_check)) t23(void) {}

// CHECK: call void %{{[a-z0-9]+}}() [[NOCF_CHECK_CALL:#[0-9]+]]
typedef void (*f_t)(void);
void t24(f_t f1) {
  __attribute__((nocf_check)) f_t p = f1;
  (*p)();
}

// CHECK: attributes [[NUW]] = { noinline nounwind{{.*}} }
// CHECK: attributes [[NR]] = { noinline noreturn nounwind{{.*}} }
// CHECK: attributes [[COLDDEF]] = { cold {{.*}}}
// CHECK: attributes [[COLDDECL]] = { cold {{.*}}}
// CHECK: attributes [[HOTDEF]] = { hot {{.*}}}
// CHECK: attributes [[HOTDECL]] = { hot {{.*}}}
// CHECK: attributes [[NOCF_CHECK_FUNC]] = { nocf_check {{.*}}}
// CHECK: attributes [[COLDSITE]] = { cold {{.*}}}
// CHECK: attributes [[HOTSITE]] = { hot {{.*}}}
// CHECK: attributes [[NOCF_CHECK_CALL]] = { nocf_check }
