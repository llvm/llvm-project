// RUN: %clang_cc1 -Wall -Werror -triple thumbv8-linux-gnueabi -fno-signed-char -disable-O0-optnone -emit-llvm -o - %s | opt -S -mem2reg | FileCheck %s
// RUN: %clang_cc1 -Wall -Werror -triple arm64-apple-ios7.0 -disable-O0-optnone -emit-llvm -o - %s | opt -S -mem2reg | FileCheck %s --check-prefix=CHECK-ARM64

struct Simple {
  char a, b;
};

int test_ldrex(char *addr, long long *addr64, float *addrfloat) {
// CHECK-LABEL: @test_ldrex
// CHECK-ARM64-LABEL: @test_ldrex
  int sum = 0;
  sum += __builtin_arm_ldrex(addr);
// CHECK: [[INTRES:%.*]] = call i32 @llvm.arm.ldrex.p0(ptr elementtype(i8) %addr)
// CHECK: trunc i32 [[INTRES]] to i8

// CHECK-ARM64: [[INTRES:%.*]] = call i64 @llvm.aarch64.ldxr.p0(ptr elementtype(i8) %addr)
// CHECK-ARM64: trunc i64 [[INTRES]] to i8

  sum += __builtin_arm_ldrex((short *)addr);
// CHECK: [[INTRES:%.*]] = call i32 @llvm.arm.ldrex.p0(ptr elementtype(i16) %addr)
// CHECK: trunc i32 [[INTRES]] to i16

// CHECK-ARM64: [[INTRES:%.*]] = call i64 @llvm.aarch64.ldxr.p0(ptr elementtype(i16) %addr)
// CHECK-ARM64: trunc i64 [[INTRES]] to i16

  sum += __builtin_arm_ldrex((int *)addr);
// CHECK: call i32 @llvm.arm.ldrex.p0(ptr elementtype(i32) %addr)

// CHECK-ARM64: [[INTRES:%.*]] = call i64 @llvm.aarch64.ldxr.p0(ptr elementtype(i32) %addr)
// CHECK-ARM64: trunc i64 [[INTRES]] to i32

  sum += __builtin_arm_ldrex((long long *)addr);
// CHECK: call { i32, i32 } @llvm.arm.ldrexd(ptr %addr)

// CHECK-ARM64: call i64 @llvm.aarch64.ldxr.p0(ptr elementtype(i64) %addr)

  sum += __builtin_arm_ldrex(addr64);
// CHECK: call { i32, i32 } @llvm.arm.ldrexd(ptr %addr64)

// CHECK-ARM64: call i64 @llvm.aarch64.ldxr.p0(ptr elementtype(i64) %addr64)

  sum += __builtin_arm_ldrex(addrfloat);
// CHECK: [[INTRES:%.*]] = call i32 @llvm.arm.ldrex.p0(ptr elementtype(i32) %addrfloat)

// CHECK-ARM64: [[INTRES:%.*]] = call i64 @llvm.aarch64.ldxr.p0(ptr elementtype(i32) %addrfloat)
// CHECK-ARM64: [[TRUNCRES:%.*]] = trunc i64 [[INTRES]] to i32
// CHECK-ARM64: bitcast i32 [[TRUNCRES]] to float

  sum += __builtin_arm_ldrex((double *)addr);
// CHECK: [[STRUCTRES:%.*]] = call { i32, i32 } @llvm.arm.ldrexd(ptr %addr)
// CHECK: [[RESHI:%.*]] = extractvalue { i32, i32 } [[STRUCTRES]], 1
// CHECK: [[RESLO:%.*]] = extractvalue { i32, i32 } [[STRUCTRES]], 0
// CHECK: [[RESHI64:%.*]] = zext i32 [[RESHI]] to i64
// CHECK: [[RESLO64:%.*]] = zext i32 [[RESLO]] to i64
// CHECK: [[RESHIHI:%.*]] = shl nuw i64 [[RESHI64]], 32
// CHECK: [[INTRES:%.*]] = or i64 [[RESHIHI]], [[RESLO64]]

// CHECK-ARM64: [[INTRES:%.*]] = call i64 @llvm.aarch64.ldxr.p0(ptr elementtype(i64) %addr)
// CHECK-ARM64: bitcast i64 [[INTRES]] to double

  sum += *__builtin_arm_ldrex((int **)addr);
// CHECK: [[INTRES:%.*]] = call i32 @llvm.arm.ldrex.p0(ptr elementtype(i32) %addr)
// CHECK: inttoptr i32 [[INTRES]] to ptr

// CHECK-ARM64: [[INTRES:%.*]] = call i64 @llvm.aarch64.ldxr.p0(ptr elementtype(i64) %addr)
// CHECK-ARM64: inttoptr i64 [[INTRES]] to ptr

  sum += __builtin_arm_ldrex((struct Simple **)addr)->a;
// CHECK: [[INTRES:%.*]] = call i32 @llvm.arm.ldrex.p0(ptr elementtype(i32) %addr)
// CHECK: inttoptr i32 [[INTRES]] to ptr

// CHECK-ARM64: [[INTRES:%.*]] = call i64 @llvm.aarch64.ldxr.p0(ptr elementtype(i64) %addr)
// CHECK-ARM64: inttoptr i64 [[INTRES]] to ptr
  return sum;
}

int test_ldaex(char *addr, long long *addr64, float *addrfloat) {
// CHECK-LABEL: @test_ldaex
// CHECK-ARM64-LABEL: @test_ldaex
  int sum = 0;
  sum += __builtin_arm_ldaex(addr);
// CHECK: [[INTRES:%.*]] = call i32 @llvm.arm.ldaex.p0(ptr elementtype(i8) %addr)
// CHECK: trunc i32 [[INTRES]] to i8

// CHECK-ARM64: [[INTRES:%.*]] = call i64 @llvm.aarch64.ldaxr.p0(ptr elementtype(i8) %addr)
// CHECK-ARM64: trunc i64 [[INTRES]] to i8

  sum += __builtin_arm_ldaex((short *)addr);
// CHECK: [[INTRES:%.*]] = call i32 @llvm.arm.ldaex.p0(ptr elementtype(i16) %addr)
// CHECK: trunc i32 [[INTRES]] to i16

// CHECK-ARM64: [[INTRES:%.*]] = call i64 @llvm.aarch64.ldaxr.p0(ptr elementtype(i16) %addr)
// CHECK-ARM64: [[TRUNCRES:%.*]] = trunc i64 [[INTRES]] to i16

  sum += __builtin_arm_ldaex((int *)addr);
// CHECK:  call i32 @llvm.arm.ldaex.p0(ptr elementtype(i32) %addr)

// CHECK-ARM64: [[INTRES:%.*]] = call i64 @llvm.aarch64.ldaxr.p0(ptr elementtype(i32) %addr)
// CHECK-ARM64: trunc i64 [[INTRES]] to i32

  sum += __builtin_arm_ldaex((long long *)addr);
// CHECK: call { i32, i32 } @llvm.arm.ldaexd(ptr %addr)

// CHECK-ARM64: call i64 @llvm.aarch64.ldaxr.p0(ptr elementtype(i64) %addr)

  sum += __builtin_arm_ldaex(addr64);
// CHECK: call { i32, i32 } @llvm.arm.ldaexd(ptr %addr64)

// CHECK-ARM64: call i64 @llvm.aarch64.ldaxr.p0(ptr elementtype(i64) %addr64)

  sum += __builtin_arm_ldaex(addrfloat);
// CHECK: [[INTRES:%.*]] = call i32 @llvm.arm.ldaex.p0(ptr elementtype(i32) %addrfloat)

// CHECK-ARM64: [[INTRES:%.*]] = call i64 @llvm.aarch64.ldaxr.p0(ptr elementtype(i32) %addrfloat)
// CHECK-ARM64: [[TRUNCRES:%.*]] = trunc i64 [[INTRES]] to i32
// CHECK-ARM64: bitcast i32 [[TRUNCRES]] to float

  sum += __builtin_arm_ldaex((double *)addr);
// CHECK: [[STRUCTRES:%.*]] = call { i32, i32 } @llvm.arm.ldaexd(ptr %addr)
// CHECK: [[RESHI:%.*]] = extractvalue { i32, i32 } [[STRUCTRES]], 1
// CHECK: [[RESLO:%.*]] = extractvalue { i32, i32 } [[STRUCTRES]], 0
// CHECK: [[RESHI64:%.*]] = zext i32 [[RESHI]] to i64
// CHECK: [[RESLO64:%.*]] = zext i32 [[RESLO]] to i64
// CHECK: [[RESHIHI:%.*]] = shl nuw i64 [[RESHI64]], 32
// CHECK: [[INTRES:%.*]] = or i64 [[RESHIHI]], [[RESLO64]]

// CHECK-ARM64: [[INTRES:%.*]] = call i64 @llvm.aarch64.ldaxr.p0(ptr elementtype(i64) %addr)
// CHECK-ARM64: bitcast i64 [[INTRES]] to double

  sum += *__builtin_arm_ldaex((int **)addr);
// CHECK: [[INTRES:%.*]] = call i32 @llvm.arm.ldaex.p0(ptr elementtype(i32) %addr)
// CHECK: inttoptr i32 [[INTRES]] to ptr

// CHECK-ARM64: [[INTRES:%.*]] = call i64 @llvm.aarch64.ldaxr.p0(ptr elementtype(i64) %addr)
// CHECK-ARM64: inttoptr i64 [[INTRES]] to ptr

  sum += __builtin_arm_ldaex((struct Simple **)addr)->a;
// CHECK: [[INTRES:%.*]] = call i32 @llvm.arm.ldaex.p0(ptr elementtype(i32) %addr)
// CHECK: inttoptr i32 [[INTRES]] to ptr

// CHECK-ARM64: [[INTRES:%.*]] = call i64 @llvm.aarch64.ldaxr.p0(ptr elementtype(i64) %addr)
// CHECK-ARM64: inttoptr i64 [[INTRES]] to ptr
  return sum;
}

int test_strex(char *addr) {
// CHECK-LABEL: @test_strex
// CHECK-ARM64-LABEL: @test_strex
  int res = 0;
  struct Simple var = {0};
  res |= __builtin_arm_strex(4, addr);
// CHECK: call i32 @llvm.arm.strex.p0(i32 4, ptr elementtype(i8) %addr)

// CHECK-ARM64: call i32 @llvm.aarch64.stxr.p0(i64 4, ptr elementtype(i8) %addr)

  res |= __builtin_arm_strex(42, (short *)addr);
// CHECK:  call i32 @llvm.arm.strex.p0(i32 42, ptr elementtype(i16) %addr)

// CHECK-ARM64:  call i32 @llvm.aarch64.stxr.p0(i64 42, ptr elementtype(i16) %addr)

  res |= __builtin_arm_strex(42, (int *)addr);
// CHECK: call i32 @llvm.arm.strex.p0(i32 42, ptr elementtype(i32) %addr)

// CHECK-ARM64: call i32 @llvm.aarch64.stxr.p0(i64 42, ptr elementtype(i32) %addr)

  res |= __builtin_arm_strex(42, (long long *)addr);
// CHECK: store i64 42, ptr [[TMP:%.*]], align 8
// CHECK: [[LOHI:%.*]] = load { i32, i32 }, ptr [[TMP]]
// CHECK: [[LO:%.*]] = extractvalue { i32, i32 } [[LOHI]], 0
// CHECK: [[HI:%.*]] = extractvalue { i32, i32 } [[LOHI]], 1
// CHECK: call i32 @llvm.arm.strexd(i32 [[LO]], i32 [[HI]], ptr %addr)

// CHECK-ARM64: call i32 @llvm.aarch64.stxr.p0(i64 42, ptr elementtype(i64) %addr)

  res |= __builtin_arm_strex(2.71828f, (float *)addr);
// CHECK: call i32 @llvm.arm.strex.p0(i32 1076754509, ptr elementtype(i32) %addr)

// CHECK-ARM64: call i32 @llvm.aarch64.stxr.p0(i64 1076754509, ptr elementtype(i32) %addr)

  res |= __builtin_arm_strex(3.14159, (double *)addr);
// CHECK: store double 3.141590e+00, ptr [[TMP:%.*]], align 8
// CHECK: [[LOHI:%.*]] = load { i32, i32 }, ptr [[TMP]]
// CHECK: [[LO:%.*]] = extractvalue { i32, i32 } [[LOHI]], 0
// CHECK: [[HI:%.*]] = extractvalue { i32, i32 } [[LOHI]], 1
// CHECK: call i32 @llvm.arm.strexd(i32 [[LO]], i32 [[HI]], ptr %addr)

// CHECK-ARM64: call i32 @llvm.aarch64.stxr.p0(i64 4614256650576692846, ptr elementtype(i64) %addr)

  res |= __builtin_arm_strex(&var, (struct Simple **)addr);
// CHECK: [[INTVAL:%.*]] = ptrtoint ptr %var to i32
// CHECK: call i32 @llvm.arm.strex.p0(i32 [[INTVAL]], ptr elementtype(i32) %addr)

// CHECK-ARM64: [[INTVAL:%.*]] = ptrtoint ptr %var to i64
// CHECK-ARM64: call i32 @llvm.aarch64.stxr.p0(i64 [[INTVAL]], ptr elementtype(i64) %addr)

  return res;
}

int test_stlex(char *addr) {
// CHECK-LABEL: @test_stlex
// CHECK-ARM64-LABEL: @test_stlex
  int res = 0;
  struct Simple var = {0};
  res |= __builtin_arm_stlex(4, addr);
// CHECK: call i32 @llvm.arm.stlex.p0(i32 4, ptr elementtype(i8) %addr)

// CHECK-ARM64: call i32 @llvm.aarch64.stlxr.p0(i64 4, ptr elementtype(i8) %addr)

  res |= __builtin_arm_stlex(42, (short *)addr);
// CHECK:  call i32 @llvm.arm.stlex.p0(i32 42, ptr elementtype(i16) %addr)

// CHECK-ARM64:  call i32 @llvm.aarch64.stlxr.p0(i64 42, ptr elementtype(i16) %addr)

  res |= __builtin_arm_stlex(42, (int *)addr);
// CHECK: call i32 @llvm.arm.stlex.p0(i32 42, ptr elementtype(i32) %addr)

// CHECK-ARM64: call i32 @llvm.aarch64.stlxr.p0(i64 42, ptr elementtype(i32) %addr)

  res |= __builtin_arm_stlex(42, (long long *)addr);
// CHECK: store i64 42, ptr [[TMP:%.*]], align 8
// CHECK: [[LOHI:%.*]] = load { i32, i32 }, ptr [[TMP]]
// CHECK: [[LO:%.*]] = extractvalue { i32, i32 } [[LOHI]], 0
// CHECK: [[HI:%.*]] = extractvalue { i32, i32 } [[LOHI]], 1
// CHECK: call i32 @llvm.arm.stlexd(i32 [[LO]], i32 [[HI]], ptr %addr)

// CHECK-ARM64: call i32 @llvm.aarch64.stlxr.p0(i64 42, ptr elementtype(i64) %addr)

  res |= __builtin_arm_stlex(2.71828f, (float *)addr);
// CHECK: call i32 @llvm.arm.stlex.p0(i32 1076754509, ptr elementtype(i32) %addr)

// CHECK-ARM64: call i32 @llvm.aarch64.stlxr.p0(i64 1076754509, ptr elementtype(i32) %addr)

  res |= __builtin_arm_stlex(3.14159, (double *)addr);
// CHECK: store double 3.141590e+00, ptr [[TMP:%.*]], align 8
// CHECK: [[LOHI:%.*]] = load { i32, i32 }, ptr [[TMP]]
// CHECK: [[LO:%.*]] = extractvalue { i32, i32 } [[LOHI]], 0
// CHECK: [[HI:%.*]] = extractvalue { i32, i32 } [[LOHI]], 1
// CHECK: call i32 @llvm.arm.stlexd(i32 [[LO]], i32 [[HI]], ptr %addr)

// CHECK-ARM64: call i32 @llvm.aarch64.stlxr.p0(i64 4614256650576692846, ptr elementtype(i64) %addr)

  res |= __builtin_arm_stlex(&var, (struct Simple **)addr);
// CHECK: [[INTVAL:%.*]] = ptrtoint ptr %var to i32
// CHECK: call i32 @llvm.arm.stlex.p0(i32 [[INTVAL]], ptr elementtype(i32) %addr)

// CHECK-ARM64: [[INTVAL:%.*]] = ptrtoint ptr %var to i64
// CHECK-ARM64: call i32 @llvm.aarch64.stlxr.p0(i64 [[INTVAL]], ptr elementtype(i64) %addr)

  return res;
}

void test_clrex(void) {
// CHECK-LABEL: @test_clrex
// CHECK-ARM64-LABEL: @test_clrex

  __builtin_arm_clrex();
// CHECK: call void @llvm.arm.clrex()
// CHECK-ARM64: call void @llvm.aarch64.clrex()
}

#ifdef __aarch64__
// 128-bit tests

__int128 test_ldrex_128(__int128 *addr) {
// CHECK-ARM64-LABEL: @test_ldrex_128

  return __builtin_arm_ldrex(addr);
// CHECK-ARM64: [[STRUCTRES:%.*]] = call { i64, i64 } @llvm.aarch64.ldxp(ptr %addr)
// CHECK-ARM64: [[RESHI:%.*]] = extractvalue { i64, i64 } [[STRUCTRES]], 1
// CHECK-ARM64: [[RESLO:%.*]] = extractvalue { i64, i64 } [[STRUCTRES]], 0
// CHECK-ARM64: [[RESHI64:%.*]] = zext i64 [[RESHI]] to i128
// CHECK-ARM64: [[RESLO64:%.*]] = zext i64 [[RESLO]] to i128
// CHECK-ARM64: [[RESHIHI:%.*]] = shl nuw i128 [[RESHI64]], 64
// CHECK-ARM64: [[INTRES:%.*]] = or i128 [[RESHIHI]], [[RESLO64]]
// CHECK-ARM64: ret i128 [[INTRES]]
}

int test_strex_128(__int128 *addr, __int128 val) {
// CHECK-ARM64-LABEL: @test_strex_128

  return __builtin_arm_strex(val, addr);
// CHECK-ARM64: store i128 %val, ptr [[TMP:%.*]], align 16
// CHECK-ARM64: [[LOHI:%.*]] = load { i64, i64 }, ptr [[TMP]]
// CHECK-ARM64: [[LO:%.*]] = extractvalue { i64, i64 } [[LOHI]], 0
// CHECK-ARM64: [[HI:%.*]] = extractvalue { i64, i64 } [[LOHI]], 1
// CHECK-ARM64: call i32 @llvm.aarch64.stxp(i64 [[LO]], i64 [[HI]], ptr %addr)
}

__int128 test_ldaex_128(__int128 *addr) {
// CHECK-ARM64-LABEL: @test_ldaex_128

  return __builtin_arm_ldaex(addr);
// CHECK-ARM64: [[STRUCTRES:%.*]] = call { i64, i64 } @llvm.aarch64.ldaxp(ptr %addr)
// CHECK-ARM64: [[RESHI:%.*]] = extractvalue { i64, i64 } [[STRUCTRES]], 1
// CHECK-ARM64: [[RESLO:%.*]] = extractvalue { i64, i64 } [[STRUCTRES]], 0
// CHECK-ARM64: [[RESHI64:%.*]] = zext i64 [[RESHI]] to i128
// CHECK-ARM64: [[RESLO64:%.*]] = zext i64 [[RESLO]] to i128
// CHECK-ARM64: [[RESHIHI:%.*]] = shl nuw i128 [[RESHI64]], 64
// CHECK-ARM64: [[INTRES:%.*]] = or i128 [[RESHIHI]], [[RESLO64]]
// CHECK-ARM64: ret i128 [[INTRES]]
}

int test_stlex_128(__int128 *addr, __int128 val) {
// CHECK-ARM64-LABEL: @test_stlex_128

  return __builtin_arm_stlex(val, addr);
// CHECK-ARM64: store i128 %val, ptr [[TMP:%.*]], align 16
// CHECK-ARM64: [[LOHI:%.*]] = load { i64, i64 }, ptr [[TMP]]
// CHECK-ARM64: [[LO:%.*]] = extractvalue { i64, i64 } [[LOHI]], 0
// CHECK-ARM64: [[HI:%.*]] = extractvalue { i64, i64 } [[LOHI]], 1
// CHECK-ARM64: [[RES:%.*]] = call i32 @llvm.aarch64.stlxp(i64 [[LO]], i64 [[HI]], ptr %addr)
}

#endif
