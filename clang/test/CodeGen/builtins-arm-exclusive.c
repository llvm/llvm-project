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
// CHECK: [[INTRES:%.*]] = call i32 @llvm.arm.ldrex.p0i8(i8* %addr)
// CHECK: trunc i32 [[INTRES]] to i8

// CHECK-ARM64: [[INTRES:%.*]] = call i64 @llvm.aarch64.ldxr.p0i8(i8* %addr)
// CHECK-ARM64: trunc i64 [[INTRES]] to i8

  sum += __builtin_arm_ldrex((short *)addr);
// CHECK: [[ADDR16:%.*]] = bitcast i8* %addr to i16*
// CHECK: [[INTRES:%.*]] = call i32 @llvm.arm.ldrex.p0i16(i16* [[ADDR16]])
// CHECK: trunc i32 [[INTRES]] to i16

// CHECK-ARM64: [[ADDR16:%.*]] = bitcast i8* %addr to i16*
// CHECK-ARM64: [[INTRES:%.*]] = call i64 @llvm.aarch64.ldxr.p0i16(i16* [[ADDR16]])
// CHECK-ARM64: trunc i64 [[INTRES]] to i16

  sum += __builtin_arm_ldrex((int *)addr);
// CHECK: [[ADDR32:%.*]] = bitcast i8* %addr to i32*
// CHECK: call i32 @llvm.arm.ldrex.p0i32(i32* [[ADDR32]])

// CHECK-ARM64: [[ADDR32:%.*]] = bitcast i8* %addr to i32*
// CHECK-ARM64: [[INTRES:%.*]] = call i64 @llvm.aarch64.ldxr.p0i32(i32* [[ADDR32]])
// CHECK-ARM64: trunc i64 [[INTRES]] to i32

  sum += __builtin_arm_ldrex((long long *)addr);
// CHECK: [[TMP4:%.*]] = bitcast i8* %addr to i64*
// CHECK: [[TMP5:%.*]] = bitcast i64* [[TMP4]] to i8*
// CHECK: call { i32, i32 } @llvm.arm.ldrexd(i8* [[TMP5]])

// CHECK-ARM64: [[ADDR64:%.*]] = bitcast i8* %addr to i64*
// CHECK-ARM64: call i64 @llvm.aarch64.ldxr.p0i64(i64* [[ADDR64]])

  sum += __builtin_arm_ldrex(addr64);
// CHECK: [[ADDR64_AS8:%.*]] = bitcast i64* %addr64 to i8*
// CHECK: call { i32, i32 } @llvm.arm.ldrexd(i8* [[ADDR64_AS8]])

// CHECK-ARM64: call i64 @llvm.aarch64.ldxr.p0i64(i64* %addr64)

  sum += __builtin_arm_ldrex(addrfloat);
// CHECK: [[INTADDR:%.*]] = bitcast float* %addrfloat to i32*
// CHECK: [[INTRES:%.*]] = call i32 @llvm.arm.ldrex.p0i32(i32* [[INTADDR]])
// CHECK: bitcast i32 [[INTRES]] to float

// CHECK-ARM64: [[INTADDR:%.*]] = bitcast float* %addrfloat to i32*
// CHECK-ARM64: [[INTRES:%.*]] = call i64 @llvm.aarch64.ldxr.p0i32(i32* [[INTADDR]])
// CHECK-ARM64: [[TRUNCRES:%.*]] = trunc i64 [[INTRES]] to i32
// CHECK-ARM64: bitcast i32 [[TRUNCRES]] to float

  sum += __builtin_arm_ldrex((double *)addr);
// CHECK: [[TMP4:%.*]] = bitcast i8* %addr to double*
// CHECK: [[TMP5:%.*]] = bitcast double* [[TMP4]] to i8*
// CHECK: [[STRUCTRES:%.*]] = call { i32, i32 } @llvm.arm.ldrexd(i8* [[TMP5]])
// CHECK: [[RESHI:%.*]] = extractvalue { i32, i32 } [[STRUCTRES]], 1
// CHECK: [[RESLO:%.*]] = extractvalue { i32, i32 } [[STRUCTRES]], 0
// CHECK: [[RESHI64:%.*]] = zext i32 [[RESHI]] to i64
// CHECK: [[RESLO64:%.*]] = zext i32 [[RESLO]] to i64
// CHECK: [[RESHIHI:%.*]] = shl nuw i64 [[RESHI64]], 32
// CHECK: [[INTRES:%.*]] = or i64 [[RESHIHI]], [[RESLO64]]
// CHECK: bitcast i64 [[INTRES]] to double

// CHECK-ARM64: [[TMP4:%.*]] = bitcast i8* %addr to double*
// CHECK-ARM64: [[TMP5:%.*]] = bitcast double* [[TMP4]] to i64*
// CHECK-ARM64: [[INTRES:%.*]] = call i64 @llvm.aarch64.ldxr.p0i64(i64* [[TMP5]])
// CHECK-ARM64: bitcast i64 [[INTRES]] to double

  sum += *__builtin_arm_ldrex((int **)addr);
// CHECK: [[TMP4:%.*]] = bitcast i8* %addr to i32**
// CHECK: [[TMP5:%.*]] = bitcast i32** [[TMP4]] to i32*
// CHECK: [[INTRES:%.*]] = call i32 @llvm.arm.ldrex.p0i32(i32* [[TMP5]])
// CHECK: inttoptr i32 [[INTRES]] to i32*

// CHECK-ARM64: [[TMP4:%.*]] = bitcast i8* %addr to i32**
// CHECK-ARM64: [[TMP5:%.*]] = bitcast i32** [[TMP4]] to i64*
// CHECK-ARM64: [[INTRES:%.*]] = call i64 @llvm.aarch64.ldxr.p0i64(i64* [[TMP5]])
// CHECK-ARM64: inttoptr i64 [[INTRES]] to i32*

  sum += __builtin_arm_ldrex((struct Simple **)addr)->a;
// CHECK: [[TMP4:%.*]] = bitcast i8* %addr to %struct.Simple**
// CHECK: [[TMP5:%.*]] = bitcast %struct.Simple** [[TMP4]] to i32*
// CHECK: [[INTRES:%.*]] = call i32 @llvm.arm.ldrex.p0i32(i32* [[TMP5]])
// CHECK: inttoptr i32 [[INTRES]] to %struct.Simple*

// CHECK-ARM64: [[TMP4:%.*]] = bitcast i8* %addr to %struct.Simple**
// CHECK-ARM64: [[TMP5:%.*]] = bitcast %struct.Simple** [[TMP4]] to i64*
// CHECK-ARM64: [[INTRES:%.*]] = call i64 @llvm.aarch64.ldxr.p0i64(i64* [[TMP5]])
// CHECK-ARM64: inttoptr i64 [[INTRES]] to %struct.Simple*
  return sum;
}

int test_ldaex(char *addr, long long *addr64, float *addrfloat) {
// CHECK-LABEL: @test_ldaex
// CHECK-ARM64-LABEL: @test_ldaex
  int sum = 0;
  sum += __builtin_arm_ldaex(addr);
// CHECK: [[INTRES:%.*]] = call i32 @llvm.arm.ldaex.p0i8(i8* %addr)
// CHECK: trunc i32 [[INTRES]] to i8

// CHECK-ARM64: [[INTRES:%.*]] = call i64 @llvm.aarch64.ldaxr.p0i8(i8* %addr)
// CHECK-ARM64: trunc i64 [[INTRES]] to i8

  sum += __builtin_arm_ldaex((short *)addr);
// CHECK: [[ADDR16:%.*]] = bitcast i8* %addr to i16*
// CHECK: [[INTRES:%.*]] = call i32 @llvm.arm.ldaex.p0i16(i16* [[ADDR16]])
// CHECK: trunc i32 [[INTRES]] to i16

// CHECK-ARM64: [[ADDR16:%.*]] = bitcast i8* %addr to i16*
// CHECK-ARM64: [[INTRES:%.*]] = call i64 @llvm.aarch64.ldaxr.p0i16(i16* [[ADDR16]])
// CHECK-ARM64: [[TRUNCRES:%.*]] = trunc i64 [[INTRES]] to i16

  sum += __builtin_arm_ldaex((int *)addr);
// CHECK: [[ADDR32:%.*]] = bitcast i8* %addr to i32*
// CHECK:  call i32 @llvm.arm.ldaex.p0i32(i32* [[ADDR32]])

// CHECK-ARM64: [[ADDR32:%.*]] = bitcast i8* %addr to i32*
// CHECK-ARM64: [[INTRES:%.*]] = call i64 @llvm.aarch64.ldaxr.p0i32(i32* [[ADDR32]])
// CHECK-ARM64: trunc i64 [[INTRES]] to i32

  sum += __builtin_arm_ldaex((long long *)addr);
// CHECK: [[TMP4:%.*]] = bitcast i8* %addr to i64*
// CHECK: [[TMP5:%.*]] = bitcast i64* [[TMP4]] to i8*
// CHECK: call { i32, i32 } @llvm.arm.ldaexd(i8* [[TMP5]])

// CHECK-ARM64: [[ADDR64:%.*]] = bitcast i8* %addr to i64*
// CHECK-ARM64: call i64 @llvm.aarch64.ldaxr.p0i64(i64* [[ADDR64]])

  sum += __builtin_arm_ldaex(addr64);
// CHECK: [[ADDR64_AS8:%.*]] = bitcast i64* %addr64 to i8*
// CHECK: call { i32, i32 } @llvm.arm.ldaexd(i8* [[ADDR64_AS8]])

// CHECK-ARM64: call i64 @llvm.aarch64.ldaxr.p0i64(i64* %addr64)

  sum += __builtin_arm_ldaex(addrfloat);
// CHECK: [[INTADDR:%.*]] = bitcast float* %addrfloat to i32*
// CHECK: [[INTRES:%.*]] = call i32 @llvm.arm.ldaex.p0i32(i32* [[INTADDR]])
// CHECK: bitcast i32 [[INTRES]] to float

// CHECK-ARM64: [[INTADDR:%.*]] = bitcast float* %addrfloat to i32*
// CHECK-ARM64: [[INTRES:%.*]] = call i64 @llvm.aarch64.ldaxr.p0i32(i32* [[INTADDR]])
// CHECK-ARM64: [[TRUNCRES:%.*]] = trunc i64 [[INTRES]] to i32
// CHECK-ARM64: bitcast i32 [[TRUNCRES]] to float

  sum += __builtin_arm_ldaex((double *)addr);
// CHECK: [[TMP4:%.*]] = bitcast i8* %addr to double*
// CHECK: [[TMP5:%.*]] = bitcast double* [[TMP4]] to i8*
// CHECK: [[STRUCTRES:%.*]] = call { i32, i32 } @llvm.arm.ldaexd(i8* [[TMP5]])
// CHECK: [[RESHI:%.*]] = extractvalue { i32, i32 } [[STRUCTRES]], 1
// CHECK: [[RESLO:%.*]] = extractvalue { i32, i32 } [[STRUCTRES]], 0
// CHECK: [[RESHI64:%.*]] = zext i32 [[RESHI]] to i64
// CHECK: [[RESLO64:%.*]] = zext i32 [[RESLO]] to i64
// CHECK: [[RESHIHI:%.*]] = shl nuw i64 [[RESHI64]], 32
// CHECK: [[INTRES:%.*]] = or i64 [[RESHIHI]], [[RESLO64]]
// CHECK: bitcast i64 [[INTRES]] to double

// CHECK-ARM64: [[TMP4:%.*]] = bitcast i8* %addr to double*
// CHECK-ARM64: [[TMP5:%.*]] = bitcast double* [[TMP4]] to i64*
// CHECK-ARM64: [[INTRES:%.*]] = call i64 @llvm.aarch64.ldaxr.p0i64(i64* [[TMP5]])
// CHECK-ARM64: bitcast i64 [[INTRES]] to double

  sum += *__builtin_arm_ldaex((int **)addr);
// CHECK: [[TMP4:%.*]] = bitcast i8* %addr to i32**
// CHECK: [[TMP5:%.*]] = bitcast i32** [[TMP4]] to i32*
// CHECK: [[INTRES:%.*]] = call i32 @llvm.arm.ldaex.p0i32(i32* [[TMP5]])
// CHECK: inttoptr i32 [[INTRES]] to i32*

// CHECK-ARM64: [[TMP4:%.*]] = bitcast i8* %addr to i32**
// CHECK-ARM64: [[TMP5:%.*]] = bitcast i32** [[TMP4]] to i64*
// CHECK-ARM64: [[INTRES:%.*]] = call i64 @llvm.aarch64.ldaxr.p0i64(i64* [[TMP5]])
// CHECK-ARM64: inttoptr i64 [[INTRES]] to i32*

  sum += __builtin_arm_ldaex((struct Simple **)addr)->a;
// CHECK: [[TMP4:%.*]] = bitcast i8* %addr to %struct.Simple**
// CHECK: [[TMP5:%.*]] = bitcast %struct.Simple** [[TMP4]] to i32*
// CHECK: [[INTRES:%.*]] = call i32 @llvm.arm.ldaex.p0i32(i32* [[TMP5]])
// CHECK: inttoptr i32 [[INTRES]] to %struct.Simple*

// CHECK-ARM64: [[TMP4:%.*]] = bitcast i8* %addr to %struct.Simple**
// CHECK-ARM64: [[TMP5:%.*]] = bitcast %struct.Simple** [[TMP4]] to i64*
// CHECK-ARM64: [[INTRES:%.*]] = call i64 @llvm.aarch64.ldaxr.p0i64(i64* [[TMP5]])
// CHECK-ARM64: inttoptr i64 [[INTRES]] to %struct.Simple*
  return sum;
}

int test_strex(char *addr) {
// CHECK-LABEL: @test_strex
// CHECK-ARM64-LABEL: @test_strex
  int res = 0;
  struct Simple var = {0};
  res |= __builtin_arm_strex(4, addr);
// CHECK: call i32 @llvm.arm.strex.p0i8(i32 4, i8* %addr)

// CHECK-ARM64: call i32 @llvm.aarch64.stxr.p0i8(i64 4, i8* %addr)

  res |= __builtin_arm_strex(42, (short *)addr);
// CHECK: [[ADDR16:%.*]] = bitcast i8* %addr to i16*
// CHECK:  call i32 @llvm.arm.strex.p0i16(i32 42, i16* [[ADDR16]])

// CHECK-ARM64: [[ADDR16:%.*]] = bitcast i8* %addr to i16*
// CHECK-ARM64:  call i32 @llvm.aarch64.stxr.p0i16(i64 42, i16* [[ADDR16]])

  res |= __builtin_arm_strex(42, (int *)addr);
// CHECK: [[ADDR32:%.*]] = bitcast i8* %addr to i32*
// CHECK: call i32 @llvm.arm.strex.p0i32(i32 42, i32* [[ADDR32]])

// CHECK-ARM64: [[ADDR32:%.*]] = bitcast i8* %addr to i32*
// CHECK-ARM64: call i32 @llvm.aarch64.stxr.p0i32(i64 42, i32* [[ADDR32]])

  res |= __builtin_arm_strex(42, (long long *)addr);
// CHECK: store i64 42, i64* [[TMP:%.*]], align 8
// CHECK: [[LOHI_ADDR:%.*]] = bitcast i64* [[TMP]] to { i32, i32 }*
// CHECK: [[LOHI:%.*]] = load { i32, i32 }, { i32, i32 }* [[LOHI_ADDR]]
// CHECK: [[LO:%.*]] = extractvalue { i32, i32 } [[LOHI]], 0
// CHECK: [[HI:%.*]] = extractvalue { i32, i32 } [[LOHI]], 1
// CHECK: [[TMP4:%.*]] = bitcast i8* %addr to i64*
// CHECK: [[TMP5:%.*]] = bitcast i64* [[TMP4]] to i8*
// CHECK: call i32 @llvm.arm.strexd(i32 [[LO]], i32 [[HI]], i8* [[TMP5]])

// CHECK-ARM64: [[ADDR64:%.*]] = bitcast i8* %addr to i64*
// CHECK-ARM64: call i32 @llvm.aarch64.stxr.p0i64(i64 42, i64* [[ADDR64]])

  res |= __builtin_arm_strex(2.71828f, (float *)addr);
// CHECK: [[TMP4:%.*]] = bitcast i8* %addr to float*
// CHECK: [[TMP5:%.*]] = bitcast float* [[TMP4]] to i32*
// CHECK: call i32 @llvm.arm.strex.p0i32(i32 1076754509, i32* [[TMP5]])

// CHECK-ARM64: [[TMP4:%.*]] = bitcast i8* %addr to float*
// CHECK-ARM64: [[TMP5:%.*]] = bitcast float* [[TMP4]] to i32*
// CHECK-ARM64: call i32 @llvm.aarch64.stxr.p0i32(i64 1076754509, i32* [[TMP5]])

  res |= __builtin_arm_strex(3.14159, (double *)addr);
// CHECK: store double 3.141590e+00, double* [[TMP:%.*]], align 8
// CHECK: [[LOHI_ADDR:%.*]] = bitcast double* [[TMP]] to { i32, i32 }*
// CHECK: [[LOHI:%.*]] = load { i32, i32 }, { i32, i32 }* [[LOHI_ADDR]]
// CHECK: [[LO:%.*]] = extractvalue { i32, i32 } [[LOHI]], 0
// CHECK: [[HI:%.*]] = extractvalue { i32, i32 } [[LOHI]], 1
// CHECK: [[TMP4:%.*]] = bitcast i8* %addr to double*
// CHECK: [[TMP5:%.*]] = bitcast double* [[TMP4]] to i8*
// CHECK: call i32 @llvm.arm.strexd(i32 [[LO]], i32 [[HI]], i8* [[TMP5]])

// CHECK-ARM64: [[TMP4:%.*]] = bitcast i8* %addr to double*
// CHECK-ARM64: [[TMP5:%.*]] = bitcast double* [[TMP4]] to i64*
// CHECK-ARM64: call i32 @llvm.aarch64.stxr.p0i64(i64 4614256650576692846, i64* [[TMP5]])

  res |= __builtin_arm_strex(&var, (struct Simple **)addr);
// CHECK: [[TMP4:%.*]] = bitcast i8* %addr to %struct.Simple**
// CHECK: [[TMP5:%.*]] = bitcast %struct.Simple** [[TMP4]] to i32*
// CHECK: [[INTVAL:%.*]] = ptrtoint %struct.Simple* %var to i32
// CHECK: call i32 @llvm.arm.strex.p0i32(i32 [[INTVAL]], i32* [[TMP5]])

// CHECK-ARM64: [[TMP4:%.*]] = bitcast i8* %addr to %struct.Simple**
// CHECK-ARM64: [[TMP5:%.*]] = bitcast %struct.Simple** [[TMP4]] to i64*
// CHECK-ARM64: [[INTVAL:%.*]] = ptrtoint %struct.Simple* %var to i64
// CHECK-ARM64: call i32 @llvm.aarch64.stxr.p0i64(i64 [[INTVAL]], i64* [[TMP5]])

  return res;
}

int test_stlex(char *addr) {
// CHECK-LABEL: @test_stlex
// CHECK-ARM64-LABEL: @test_stlex
  int res = 0;
  struct Simple var = {0};
  res |= __builtin_arm_stlex(4, addr);
// CHECK: call i32 @llvm.arm.stlex.p0i8(i32 4, i8* %addr)

// CHECK-ARM64: call i32 @llvm.aarch64.stlxr.p0i8(i64 4, i8* %addr)

  res |= __builtin_arm_stlex(42, (short *)addr);
// CHECK: [[ADDR16:%.*]] = bitcast i8* %addr to i16*
// CHECK:  call i32 @llvm.arm.stlex.p0i16(i32 42, i16* [[ADDR16]])

// CHECK-ARM64: [[ADDR16:%.*]] = bitcast i8* %addr to i16*
// CHECK-ARM64:  call i32 @llvm.aarch64.stlxr.p0i16(i64 42, i16* [[ADDR16]])

  res |= __builtin_arm_stlex(42, (int *)addr);
// CHECK: [[ADDR32:%.*]] = bitcast i8* %addr to i32*
// CHECK: call i32 @llvm.arm.stlex.p0i32(i32 42, i32* [[ADDR32]])

// CHECK-ARM64: [[ADDR32:%.*]] = bitcast i8* %addr to i32*
// CHECK-ARM64: call i32 @llvm.aarch64.stlxr.p0i32(i64 42, i32* [[ADDR32]])

  res |= __builtin_arm_stlex(42, (long long *)addr);
// CHECK: store i64 42, i64* [[TMP:%.*]], align 8
// CHECK: [[LOHI_ADDR:%.*]] = bitcast i64* [[TMP]] to { i32, i32 }*
// CHECK: [[LOHI:%.*]] = load { i32, i32 }, { i32, i32 }* [[LOHI_ADDR]]
// CHECK: [[LO:%.*]] = extractvalue { i32, i32 } [[LOHI]], 0
// CHECK: [[HI:%.*]] = extractvalue { i32, i32 } [[LOHI]], 1
// CHECK: [[TMP4:%.*]] = bitcast i8* %addr to i64*
// CHECK: [[TMP5:%.*]] = bitcast i64* [[TMP4]] to i8*
// CHECK: call i32 @llvm.arm.stlexd(i32 [[LO]], i32 [[HI]], i8* [[TMP5]])

// CHECK-ARM64: [[ADDR64:%.*]] = bitcast i8* %addr to i64*
// CHECK-ARM64: call i32 @llvm.aarch64.stlxr.p0i64(i64 42, i64* [[ADDR64]])

  res |= __builtin_arm_stlex(2.71828f, (float *)addr);
// CHECK: [[TMP4:%.*]] = bitcast i8* %addr to float*
// CHECK: [[TMP5:%.*]] = bitcast float* [[TMP4]] to i32*
// CHECK: call i32 @llvm.arm.stlex.p0i32(i32 1076754509, i32* [[TMP5]])

// CHECK-ARM64: [[TMP4:%.*]] = bitcast i8* %addr to float*
// CHECK-ARM64: [[TMP5:%.*]] = bitcast float* [[TMP4]] to i32*
// CHECK-ARM64: call i32 @llvm.aarch64.stlxr.p0i32(i64 1076754509, i32* [[TMP5]])

  res |= __builtin_arm_stlex(3.14159, (double *)addr);
// CHECK: store double 3.141590e+00, double* [[TMP:%.*]], align 8
// CHECK: [[LOHI_ADDR:%.*]] = bitcast double* [[TMP]] to { i32, i32 }*
// CHECK: [[LOHI:%.*]] = load { i32, i32 }, { i32, i32 }* [[LOHI_ADDR]]
// CHECK: [[LO:%.*]] = extractvalue { i32, i32 } [[LOHI]], 0
// CHECK: [[HI:%.*]] = extractvalue { i32, i32 } [[LOHI]], 1
// CHECK: [[TMP4:%.*]] = bitcast i8* %addr to double*
// CHECK: [[TMP5:%.*]] = bitcast double* [[TMP4]] to i8*
// CHECK: call i32 @llvm.arm.stlexd(i32 [[LO]], i32 [[HI]], i8* [[TMP5]])

// CHECK-ARM64: [[TMP4:%.*]] = bitcast i8* %addr to double*
// CHECK-ARM64: [[TMP5:%.*]] = bitcast double* [[TMP4]] to i64*
// CHECK-ARM64: call i32 @llvm.aarch64.stlxr.p0i64(i64 4614256650576692846, i64* [[TMP5]])

  res |= __builtin_arm_stlex(&var, (struct Simple **)addr);
// CHECK: [[TMP4:%.*]] = bitcast i8* %addr to %struct.Simple**
// CHECK: [[TMP5:%.*]] = bitcast %struct.Simple** [[TMP4]] to i32*
// CHECK: [[INTVAL:%.*]] = ptrtoint %struct.Simple* %var to i32
// CHECK: call i32 @llvm.arm.stlex.p0i32(i32 [[INTVAL]], i32* [[TMP5]])

// CHECK-ARM64: [[TMP4:%.*]] = bitcast i8* %addr to %struct.Simple**
// CHECK-ARM64: [[TMP5:%.*]] = bitcast %struct.Simple** [[TMP4]] to i64*
// CHECK-ARM64: [[INTVAL:%.*]] = ptrtoint %struct.Simple* %var to i64
// CHECK-ARM64: call i32 @llvm.aarch64.stlxr.p0i64(i64 [[INTVAL]], i64* [[TMP5]])

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
// CHECK-ARM64: [[ADDR8:%.*]] = bitcast i128* %addr to i8*
// CHECK-ARM64: [[STRUCTRES:%.*]] = call { i64, i64 } @llvm.aarch64.ldxp(i8* [[ADDR8]])
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
// CHECK-ARM64: store i128 %val, i128* [[TMP:%.*]], align 16
// CHECK-ARM64: [[LOHI_ADDR:%.*]] = bitcast i128* [[TMP]] to { i64, i64 }*
// CHECK-ARM64: [[LOHI:%.*]] = load { i64, i64 }, { i64, i64 }* [[LOHI_ADDR]]
// CHECK-ARM64: [[LO:%.*]] = extractvalue { i64, i64 } [[LOHI]], 0
// CHECK-ARM64: [[HI:%.*]] = extractvalue { i64, i64 } [[LOHI]], 1
// CHECK-ARM64: [[ADDR8:%.*]] = bitcast i128* %addr to i8*
// CHECK-ARM64: call i32 @llvm.aarch64.stxp(i64 [[LO]], i64 [[HI]], i8* [[ADDR8]])
}

__int128 test_ldaex_128(__int128 *addr) {
// CHECK-ARM64-LABEL: @test_ldaex_128

  return __builtin_arm_ldaex(addr);
// CHECK-ARM64: [[ADDR8:%.*]] = bitcast i128* %addr to i8*
// CHECK-ARM64: [[STRUCTRES:%.*]] = call { i64, i64 } @llvm.aarch64.ldaxp(i8* [[ADDR8]])
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
// CHECK-ARM64: store i128 %val, i128* [[TMP:%.*]], align 16
// CHECK-ARM64: [[LOHI_ADDR:%.*]] = bitcast i128* [[TMP]] to { i64, i64 }*
// CHECK-ARM64: [[LOHI:%.*]] = load { i64, i64 }, { i64, i64 }* [[LOHI_ADDR]]
// CHECK-ARM64: [[LO:%.*]] = extractvalue { i64, i64 } [[LOHI]], 0
// CHECK-ARM64: [[HI:%.*]] = extractvalue { i64, i64 } [[LOHI]], 1
// CHECK-ARM64: [[ADDR8:%.*]] = bitcast i128* %addr to i8*
// CHECK-ARM64: [[RES:%.*]] = call i32 @llvm.aarch64.stlxp(i64 [[LO]], i64 [[HI]], i8* [[ADDR8]])
}

#endif
