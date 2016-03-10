// RUN: %clang_cc1 -Wall -Werror -triple thumbv7-eabi -target-cpu cortex-a8 -emit-llvm -o - %s | opt -S -mem2reg | FileCheck %s

void *f0()
{
  return __builtin_thread_pointer();
}

void f1(char *a, char *b) {
	__clear_cache(a,b);
}

// CHECK: call {{.*}} @__clear_cache

void test_eh_return_data_regno()
{
  volatile int res;
  res = __builtin_eh_return_data_regno(0);  // CHECK: store volatile i32 0
  res = __builtin_eh_return_data_regno(1);  // CHECK: store volatile i32 1
}

void nop() {
  __builtin_arm_nop();
}

// CHECK: call {{.*}} @llvm.arm.hint(i32 0)

void yield() {
  __builtin_arm_yield();
}

// CHECK: call {{.*}} @llvm.arm.hint(i32 1)

void wfe() {
  __builtin_arm_wfe();
}

// CHECK: call {{.*}} @llvm.arm.hint(i32 2)

void wfi() {
  __builtin_arm_wfi();
}

// CHECK: call {{.*}} @llvm.arm.hint(i32 3)

void sev() {
  __builtin_arm_sev();
}

// CHECK: call {{.*}} @llvm.arm.hint(i32 4)

void sevl() {
  __builtin_arm_sevl();
}

// CHECK: call {{.*}} @llvm.arm.hint(i32 5)

void dbg() {
  __builtin_arm_dbg(0);
}

// CHECK: call {{.*}} @llvm.arm.dbg(i32 0)

void test_barrier() {
  __builtin_arm_dmb(1); //CHECK: call {{.*}} @llvm.arm.dmb(i32 1)
  __builtin_arm_dsb(2); //CHECK: call {{.*}} @llvm.arm.dsb(i32 2)
  __builtin_arm_isb(3); //CHECK: call {{.*}} @llvm.arm.isb(i32 3)
}

// CHECK: call {{.*}} @llvm.arm.rbit(i32 %a)

unsigned rbit(unsigned a) {
  return __builtin_arm_rbit(a);
}

void prefetch(int i) {
  __builtin_arm_prefetch(&i, 0, 1);
// CHECK: call {{.*}} @llvm.prefetch(i8* %{{.*}}, i32 0, i32 3, i32 1)

  __builtin_arm_prefetch(&i, 1, 1);
// CHECK: call {{.*}} @llvm.prefetch(i8* %{{.*}}, i32 1, i32 3, i32 1)


  __builtin_arm_prefetch(&i, 1, 0);
// CHECK: call {{.*}} @llvm.prefetch(i8* %{{.*}}, i32 1, i32 3, i32 0)
}

unsigned mrc() {
  // CHECK: define i32 @mrc()
  // CHECK: [[R:%.*]] = call i32 @llvm.arm.mrc(i32 15, i32 0, i32 13, i32 0, i32 3)
  // CHECK-NEXT: ret i32 [[R]]
  return __builtin_arm_mrc(15, 0, 13, 0, 3);
}

unsigned mrc2() {
  // CHECK: define i32 @mrc2()
  // CHECK: [[R:%.*]] = call i32 @llvm.arm.mrc2(i32 15, i32 0, i32 13, i32 0, i32 3)
  // CHECK-NEXT: ret i32 [[R]]
  return __builtin_arm_mrc2(15, 0, 13, 0, 3);
}

void mcr(unsigned a) {
  // CHECK: define void @mcr(i32 [[A:%.*]])
  // CHECK: call void @llvm.arm.mcr(i32 15, i32 0, i32 [[A]], i32 13, i32 0, i32 3)
  __builtin_arm_mcr(15, 0, a, 13, 0, 3);
}

void mcr2(unsigned a) {
  // CHECK: define void @mcr2(i32 [[A:%.*]])
  // CHECK: call void @llvm.arm.mcr2(i32 15, i32 0, i32 [[A]], i32 13, i32 0, i32 3)
  __builtin_arm_mcr2(15, 0, a, 13, 0, 3);
}

void mcrr(unsigned a, unsigned b) {
  // CHECK: define void @mcrr(i32 [[A:%.*]], i32 [[B:%.*]])
  // CHECK: call void @llvm.arm.mcrr(i32 15, i32 0, i32 [[A]], i32 [[B]], i32 0)
  __builtin_arm_mcrr(15, 0, a, b, 0);
}

void mcrr2(unsigned a, unsigned b) {
  // CHECK: define void @mcrr2(i32 [[A:%.*]], i32 [[B:%.*]])
  // CHECK: call void @llvm.arm.mcrr2(i32 15, i32 0, i32 [[A]], i32 [[B]], i32 0)
  __builtin_arm_mcrr2(15, 0, a, b, 0);
}

unsigned rsr() {
  // CHECK: [[V0:[%A-Za-z0-9.]+]] = call i32 @llvm.read_register.i32(metadata ![[M0:.*]])
  // CHECK-NEXT: ret i32 [[V0]]
  return __builtin_arm_rsr("cp1:2:c3:c4:5");
}

unsigned long long rsr64() {
  // CHECK: [[V0:[%A-Za-z0-9.]+]] = call i64 @llvm.read_register.i64(metadata ![[M1:.*]])
  // CHECK-NEXT: ret i64 [[V0]]
  return __builtin_arm_rsr64("cp1:2:c3");
}

void *rsrp() {
  // CHECK: [[V0:[%A-Za-z0-9.]+]] = call i32 @llvm.read_register.i32(metadata ![[M2:.*]])
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

// CHECK: ![[M0]] = !{!"cp1:2:c3:c4:5"}
// CHECK: ![[M1]] = !{!"cp1:2:c3"}
// CHECK: ![[M2]] = !{!"sysreg"}
