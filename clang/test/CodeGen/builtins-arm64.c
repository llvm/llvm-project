// RUN: %clang_cc1 -triple arm64-unknown-linux -disable-O0-optnone -emit-llvm -o - %s | opt -S -mem2reg | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-LINUX
// RUN: %clang_cc1 -triple aarch64-windows -disable-O0-optnone -S -emit-llvm -o - %s | opt -S -mem2reg | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-WIN
// RUN: %clang_cc1 -triple arm64_32-apple-ios13 -disable-O0-optnone -emit-llvm -o - %s | opt -S -mem2reg | FileCheck %s
#include <stdint.h>

void f0(void *a, void *b) {
	__clear_cache(a,b);
// CHECK: call {{.*}} @__clear_cache
}

void *tp (void) {
  return __builtin_thread_pointer ();
// CHECK-LINUX: call {{.*}} @llvm.thread.pointer()
}

// CHECK: call {{.*}} @llvm.bitreverse.i32(i32 %a)
unsigned rbit(unsigned a) {
  return __builtin_arm_rbit(a);
}

// CHECK-WIN: [[A64:%[^ ]+]] = zext i32 %a to i64
// CHECK-WIN: call i64 @llvm.bitreverse.i64(i64 [[A64]])
// CHECK-LINUX: call i64 @llvm.bitreverse.i64(i64 %a)
unsigned long rbitl(unsigned long a) {
  return __builtin_arm_rbit64(a);
}

// CHECK: call {{.*}} @llvm.bitreverse.i64(i64 %a)
uint64_t rbit64(uint64_t a) {
  return __builtin_arm_rbit64(a);
}

void hints(void) {
  __builtin_arm_nop();    //CHECK: call {{.*}} @llvm.aarch64.hint(i32 0)
  __builtin_arm_yield();  //CHECK: call {{.*}} @llvm.aarch64.hint(i32 1)
  __builtin_arm_wfe();    //CHECK: call {{.*}} @llvm.aarch64.hint(i32 2)
  __builtin_arm_wfi();    //CHECK: call {{.*}} @llvm.aarch64.hint(i32 3)
  __builtin_arm_sev();    //CHECK: call {{.*}} @llvm.aarch64.hint(i32 4)
  __builtin_arm_sevl();   //CHECK: call {{.*}} @llvm.aarch64.hint(i32 5)
}

void barriers(void) {
  __builtin_arm_dmb(1);  //CHECK: call {{.*}} @llvm.aarch64.dmb(i32 1)
  __builtin_arm_dsb(2);  //CHECK: call {{.*}} @llvm.aarch64.dsb(i32 2)
  __builtin_arm_isb(3);  //CHECK: call {{.*}} @llvm.aarch64.isb(i32 3)
}

void prefetch(void) {
  __builtin_arm_prefetch(0, 1, 2, 0, 1); // pstl3keep
  // CHECK: call {{.*}} @llvm.prefetch.p0(ptr null, i32 1, i32 1, i32 1)

  __builtin_arm_prefetch(0, 0, 0, 1, 1); // pldl1keep
  // CHECK: call {{.*}} @llvm.prefetch.p0(ptr null, i32 0, i32 0, i32 1)

  __builtin_arm_prefetch(0, 0, 0, 1, 1); // pldl1strm
  // CHECK: call {{.*}} @llvm.prefetch.p0(ptr null, i32 0, i32 0, i32 1)

  __builtin_arm_prefetch(0, 0, 0, 0, 0); // plil1keep
  // CHECK: call {{.*}} @llvm.prefetch.p0(ptr null, i32 0, i32 3, i32 0)
}

__attribute__((target("v8.5a")))
int32_t jcvt(double v) {
  //CHECK-LABEL: @jcvt(
  //CHECK: call i32 @llvm.aarch64.fjcvtzs
  return __builtin_arm_jcvt(v);
}

__typeof__(__builtin_arm_rsr("1:2:3:4:5")) rsr(void);

uint32_t rsr(void) {
  // CHECK: [[V0:[%A-Za-z0-9.]+]] = call i64 @llvm.read_volatile_register.i64(metadata ![[M0:[0-9]]])
  // CHECK-NEXT: trunc i64 [[V0]] to i32
  return __builtin_arm_rsr("1:2:3:4:5");
}

__typeof__(__builtin_arm_rsr64("1:2:3:4:5")) rsr64(void);

uint64_t rsr64(void) {
  // CHECK: call i64 @llvm.read_volatile_register.i64(metadata ![[M0:[0-9]]])
  return __builtin_arm_rsr64("1:2:3:4:5");
}

void *rsrp(void) {
  // CHECK: [[V0:[%A-Za-z0-9.]+]] = call i64 @llvm.read_volatile_register.i64(metadata ![[M0:[0-9]]])
  // CHECK-NEXT: inttoptr i64 [[V0]] to ptr
  return __builtin_arm_rsrp("1:2:3:4:5");
}

__typeof__(__builtin_arm_wsr("1:2:3:4:5", 0)) wsr(unsigned);

void wsr(unsigned v) {
  // CHECK: [[V0:[%A-Za-z0-9.]+]] = zext i32 %v to i64
  // CHECK-NEXT: call void @llvm.write_register.i64(metadata ![[M0:[0-9]]], i64 [[V0]])
  __builtin_arm_wsr("1:2:3:4:5", v);
}

__typeof__(__builtin_arm_wsr64("1:2:3:4:5", 0)) wsr64(uint64_t);

void wsr64(uint64_t v) {
  // CHECK: call void @llvm.write_register.i64(metadata ![[M0:[0-9]]], i64 %v)
  __builtin_arm_wsr64("1:2:3:4:5", v);
}

void wsrp(void *v) {
  // CHECK: [[V0:[%A-Za-z0-9.]+]] = ptrtoint ptr %v to i64
  // CHECK-NEXT: call void @llvm.write_register.i64(metadata ![[M0:[0-9]]], i64 [[V0]])
  __builtin_arm_wsrp("1:2:3:4:5", v);
}

unsigned int cls(uint32_t v) {
  // CHECK: call i32 @llvm.aarch64.cls(i32 %v)
  return __builtin_arm_cls(v);
}

unsigned int clsl(unsigned long v) {
  // CHECK-WIN: [[V64:%[^ ]+]] = zext i32 %v to i64
  // CHECK-WIN: call i32 @llvm.aarch64.cls64(i64 [[V64]]
  // CHECK-LINUX: call i32 @llvm.aarch64.cls64(i64 %v)
  return __builtin_arm_cls64(v);
}

unsigned int clsll(uint64_t v) {
  // CHECK: call i32 @llvm.aarch64.cls64(i64 %v)
  return __builtin_arm_cls64(v);
}

// CHECK-LABEL: @rndr(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = call { i64, i1 } @llvm.aarch64.rndr()
// CHECK-NEXT:    [[TMP1:%.*]] = extractvalue { i64, i1 } [[TMP0]], 0
// CHECK-NEXT:    [[TMP2:%.*]] = extractvalue { i64, i1 } [[TMP0]], 1
// CHECK-NEXT:    store i64 [[TMP1]], ptr [[__ADDR:%.*]], align 8
// CHECK-NEXT:    [[TMP3:%.*]] = zext i1 [[TMP2]] to i32
// CHECK-NEXT:    ret i32 [[TMP3]]
//
__attribute__((target("rand")))
int rndr(uint64_t *__addr) {
  return __builtin_arm_rndr(__addr);
}

// CHECK-LABEL: @rndrrs(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = call { i64, i1 } @llvm.aarch64.rndrrs()
// CHECK-NEXT:    [[TMP1:%.*]] = extractvalue { i64, i1 } [[TMP0]], 0
// CHECK-NEXT:    [[TMP2:%.*]] = extractvalue { i64, i1 } [[TMP0]], 1
// CHECK-NEXT:    store i64 [[TMP1]], ptr [[__ADDR:%.*]], align 8
// CHECK-NEXT:    [[TMP3:%.*]] = zext i1 [[TMP2]] to i32
// CHECK-NEXT:    ret i32 [[TMP3]]
//
__attribute__((target("rand")))
int rndrrs(uint64_t *__addr) {
  return __builtin_arm_rndrrs(__addr);
}

// CHECK: ![[M0]] = !{!"1:2:3:4:5"}
