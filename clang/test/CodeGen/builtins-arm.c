// RUN: %clang_cc1 -no-opaque-pointers -Wall -Wno-unused-but-set-variable -Werror -triple thumbv7-eabi -target-cpu cortex-a8 -disable-O0-optnone -emit-llvm -o - %s | opt -S -mem2reg | FileCheck %s

#include <stdint.h>

void *f0(void)
{
  return __builtin_thread_pointer();
}

void f1(char *a, char *b) {
  // CHECK: call {{.*}} @__clear_cache
	__clear_cache(a,b);
}

float test_vcvtrf0(float f) {
  // CHECK: call float @llvm.arm.vcvtr.f32(float %f)
  return __builtin_arm_vcvtr_f(f, 0);
}

float test_vcvtrf1(float f) {
  // CHECK: call float @llvm.arm.vcvtru.f32(float %f)
  return __builtin_arm_vcvtr_f(f, 1);
}

double test_vcvtrd0(double d) {
  // CHECK: call float @llvm.arm.vcvtr.f64(double %d)
  return __builtin_arm_vcvtr_d(d, 0);
}

double test_vcvtrd1(double d) {
  // call float @llvm.arm.vcvtru.f64(double %d)
  return __builtin_arm_vcvtr_d(d, 1);
}

void test_eh_return_data_regno(void)
{
  // CHECK: store volatile i32 0
  // CHECK: store volatile i32 1
  volatile int res;
  res = __builtin_eh_return_data_regno(0);
  res = __builtin_eh_return_data_regno(1);
}

void nop(void) {
  // CHECK: call {{.*}} @llvm.arm.hint(i32 0)
  __builtin_arm_nop();
}

void yield(void) {
  // CHECK: call {{.*}} @llvm.arm.hint(i32 1)
  __builtin_arm_yield();
}

void wfe(void) {
  // CHECK: call {{.*}} @llvm.arm.hint(i32 2)
  __builtin_arm_wfe();
}

void wfi(void) {
  // CHECK: call {{.*}} @llvm.arm.hint(i32 3)
  __builtin_arm_wfi();
}

void sev(void) {
  // CHECK: call {{.*}} @llvm.arm.hint(i32 4)
  __builtin_arm_sev();
}

void sevl(void) {
  // CHECK: call {{.*}} @llvm.arm.hint(i32 5)
  __builtin_arm_sevl();
}

void dbg(void) {
  // CHECK: call {{.*}} @llvm.arm.dbg(i32 0)
  __builtin_arm_dbg(0);
}

void test_barrier(void) {
  //CHECK: call {{.*}} @llvm.arm.dmb(i32 1)
  //CHECK: call {{.*}} @llvm.arm.dsb(i32 2)
  //CHECK: call {{.*}} @llvm.arm.isb(i32 3)
  __builtin_arm_dmb(1);
  __builtin_arm_dsb(2);
  __builtin_arm_isb(3);
}

unsigned rbit(unsigned a) {
  // CHECK: call {{.*}} @llvm.bitreverse.i32(i32 %a)
  return __builtin_arm_rbit(a);
}

void prefetch(int i) {
  __builtin_arm_prefetch(&i, 0, 1);
  // CHECK: call {{.*}} @llvm.prefetch.p0i8(i8* %{{.*}}, i32 0, i32 3, i32 1)

  __builtin_arm_prefetch(&i, 1, 1);
  // CHECK: call {{.*}} @llvm.prefetch.p0i8(i8* %{{.*}}, i32 1, i32 3, i32 1)

  __builtin_arm_prefetch(&i, 1, 0);
  // CHECK: call {{.*}} @llvm.prefetch.p0i8(i8* %{{.*}}, i32 1, i32 3, i32 0)
}

void ldc(const void *i) {
  // CHECK: define{{.*}} void @ldc(i8* noundef %i)
  // CHECK: call void @llvm.arm.ldc(i32 1, i32 2, i8* %i)
  // CHECK-NEXT: ret void
  __builtin_arm_ldc(1, 2, i);
}

void ldcl(const void *i) {
  // CHECK: define{{.*}} void @ldcl(i8* noundef %i)
  // CHECK: call void @llvm.arm.ldcl(i32 1, i32 2, i8* %i)
  // CHECK-NEXT: ret void
  __builtin_arm_ldcl(1, 2, i);
}

void ldc2(const void *i) {
  // CHECK: define{{.*}} void @ldc2(i8* noundef %i)
  // CHECK: call void @llvm.arm.ldc2(i32 1, i32 2, i8* %i)
  // CHECK-NEXT: ret void
  __builtin_arm_ldc2(1, 2, i);
}

void ldc2l(const void *i) {
  // CHECK: define{{.*}} void @ldc2l(i8* noundef %i)
  // CHECK: call void @llvm.arm.ldc2l(i32 1, i32 2, i8* %i)
  // CHECK-NEXT: ret void
  __builtin_arm_ldc2l(1, 2, i);
}

void stc(void *i) {
  // CHECK: define{{.*}} void @stc(i8* noundef %i)
  // CHECK: call void @llvm.arm.stc(i32 1, i32 2, i8* %i)
  // CHECK-NEXT: ret void
  __builtin_arm_stc(1, 2, i);
}

void stcl(void *i) {
  // CHECK: define{{.*}} void @stcl(i8* noundef %i)
  // CHECK: call void @llvm.arm.stcl(i32 1, i32 2, i8* %i)
  // CHECK-NEXT: ret void
  __builtin_arm_stcl(1, 2, i);
}

void stc2(void *i) {
  // CHECK: define{{.*}} void @stc2(i8* noundef %i)
  // CHECK: call void @llvm.arm.stc2(i32 1, i32 2, i8* %i)
  // CHECK-NEXT: ret void
  __builtin_arm_stc2(1, 2, i);
}

void stc2l(void *i) {
  // CHECK: define{{.*}} void @stc2l(i8* noundef %i)
  // CHECK: call void @llvm.arm.stc2l(i32 1, i32 2, i8* %i)
  // CHECK-NEXT: ret void
  __builtin_arm_stc2l(1, 2, i);
}

void cdp(void) {
  // CHECK: define{{.*}} void @cdp()
  // CHECK: call void @llvm.arm.cdp(i32 1, i32 2, i32 3, i32 4, i32 5, i32 6)
  // CHECK-NEXT: ret void
  __builtin_arm_cdp(1, 2, 3, 4, 5, 6);
}

void cdp2(void) {
  // CHECK: define{{.*}} void @cdp2()
  // CHECK: call void @llvm.arm.cdp2(i32 1, i32 2, i32 3, i32 4, i32 5, i32 6)
  // CHECK-NEXT: ret void
  __builtin_arm_cdp2(1, 2, 3, 4, 5, 6);
}

unsigned mrc(void) {
  // CHECK: define{{.*}} i32 @mrc()
  // CHECK: [[R:%.*]] = call i32 @llvm.arm.mrc(i32 15, i32 0, i32 13, i32 0, i32 3)
  // CHECK-NEXT: ret i32 [[R]]
  return __builtin_arm_mrc(15, 0, 13, 0, 3);
}

unsigned mrc2(void) {
  // CHECK: define{{.*}} i32 @mrc2()
  // CHECK: [[R:%.*]] = call i32 @llvm.arm.mrc2(i32 15, i32 0, i32 13, i32 0, i32 3)
  // CHECK-NEXT: ret i32 [[R]]
  return __builtin_arm_mrc2(15, 0, 13, 0, 3);
}

void mcr(unsigned a) {
  // CHECK: define{{.*}} void @mcr(i32 noundef [[A:%.*]])
  // CHECK: call void @llvm.arm.mcr(i32 15, i32 0, i32 [[A]], i32 13, i32 0, i32 3)
  __builtin_arm_mcr(15, 0, a, 13, 0, 3);
}

void mcr2(unsigned a) {
  // CHECK: define{{.*}} void @mcr2(i32 noundef [[A:%.*]])
  // CHECK: call void @llvm.arm.mcr2(i32 15, i32 0, i32 [[A]], i32 13, i32 0, i32 3)
  __builtin_arm_mcr2(15, 0, a, 13, 0, 3);
}

void mcrr(uint64_t a) {
  // CHECK: define{{.*}} void @mcrr(i64 noundef %{{.*}})
  // CHECK: call void @llvm.arm.mcrr(i32 15, i32 0, i32 %{{[0-9]+}}, i32 %{{[0-9]+}}, i32 0)
  __builtin_arm_mcrr(15, 0, a, 0);
}

void mcrr2(uint64_t a) {
  // CHECK: define{{.*}} void @mcrr2(i64 noundef %{{.*}})
  // CHECK: call void @llvm.arm.mcrr2(i32 15, i32 0, i32 %{{[0-9]+}}, i32 %{{[0-9]+}}, i32 0)
  __builtin_arm_mcrr2(15, 0, a, 0);
}

uint64_t mrrc(void) {
  // CHECK: define{{.*}} i64 @mrrc()
  // CHECK: call { i32, i32 } @llvm.arm.mrrc(i32 15, i32 0, i32 0)
  return __builtin_arm_mrrc(15, 0, 0);
}

uint64_t mrrc2(void) {
  // CHECK: define{{.*}} i64 @mrrc2()
  // CHECK: call { i32, i32 } @llvm.arm.mrrc2(i32 15, i32 0, i32 0)
  return __builtin_arm_mrrc2(15, 0, 0);
}

unsigned rsr(void) {
  // CHECK: [[V0:[%A-Za-z0-9.]+]] = call i32 @llvm.read_volatile_register.i32(metadata ![[M0:.*]])
  // CHECK-NEXT: ret i32 [[V0]]
  return __builtin_arm_rsr("cp1:2:c3:c4:5");
}

unsigned long long rsr64(void) {
  // CHECK: [[V0:[%A-Za-z0-9.]+]] = call i64 @llvm.read_volatile_register.i64(metadata ![[M1:.*]])
  // CHECK-NEXT: ret i64 [[V0]]
  return __builtin_arm_rsr64("cp1:2:c3");
}

void *rsrp(void) {
  // CHECK: [[V0:[%A-Za-z0-9.]+]] = call i32 @llvm.read_volatile_register.i32(metadata ![[M2:.*]])
  // CHECK-NEXT: [[V1:[%A-Za-z0-9.]+]] = inttoptr i32 [[V0]] to i8*
  // CHECK-NEXT: ret i8* [[V1]]
  return __builtin_arm_rsrp("sysreg");
}

void wsr(unsigned v) {
  // CHECK: call void @llvm.write_register.i32(metadata ![[M0]], i32 %v)
  __builtin_arm_wsr("cp1:2:c3:c4:5", v);
}

void wsr64(unsigned long long v) {
  // CHECK: call void @llvm.write_register.i64(metadata ![[M1]], i64 %v)
  __builtin_arm_wsr64("cp1:2:c3", v);
}

void wsrp(void *v) {
  // CHECK: [[V0:[%A-Za-z0-9.]+]] = ptrtoint i8* %v to i32
  // CHECK-NEXT: call void @llvm.write_register.i32(metadata ![[M2]], i32 [[V0]])
  __builtin_arm_wsrp("sysreg", v);
}

unsigned int cls(uint32_t v) {
  // CHECK: call i32 @llvm.arm.cls(i32 %v)
  return __builtin_arm_cls(v);
}

unsigned int clsl(unsigned long v) {
  // CHECK: call i32 @llvm.arm.cls(i32 %v)
  return __builtin_arm_cls(v);
}

unsigned int clsll(uint64_t v) {
  // CHECK: call i32 @llvm.arm.cls64(i64 %v)
  return __builtin_arm_cls64(v);
}

// CHECK: ![[M0]] = !{!"cp1:2:c3:c4:5"}
// CHECK: ![[M1]] = !{!"cp1:2:c3"}
// CHECK: ![[M2]] = !{!"sysreg"}
