// RUN: %clang_cc1 -triple arm64-windows -Wno-implicit-function-declaration -fms-compatibility -emit-llvm -o - %s \
// RUN:    | FileCheck %s --check-prefix=CHECK-MSVC --check-prefix=CHECK-MSCOMPAT

// RUN: not %clang_cc1 -triple arm64-linux -Werror -S -o /dev/null %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-LINUX

// RUN: %clang_cc1 -triple arm64-darwin -Wno-implicit-function-declaration -fms-compatibility -emit-llvm -o - -DARM64_DARWIN %s \
// RUN:    | FileCheck %s -check-prefix CHECK-MSCOMPAT

// For some reason '_InterlockedAdd` on arm64-darwin takes an 'int*' rather than a 'long*'.
#ifdef ARM64_DARWIN
typedef int int32_t;
#else
typedef long int32_t;
#endif

long test_InterlockedAdd(int32_t volatile *Addend, long Value) {
  return _InterlockedAdd(Addend, Value);
}

long test_InterlockedAdd_constant(int32_t volatile *Addend) {
  return _InterlockedAdd(Addend, -1);
}

// CHECK-LABEL: define {{.*}} i32 @test_InterlockedAdd(ptr %Addend, i32 %Value) {{.*}} {
// CHECK-MSVC: %[[OLDVAL:[0-9]+]] = atomicrmw add ptr %2, i32 %1 seq_cst, align 4
// CHECK-MSVC: %[[NEWVAL:[0-9]+]] = add i32 %[[OLDVAL:[0-9]+]], %1
// CHECK-MSVC: ret i32 %[[NEWVAL:[0-9]+]]
// CHECK-LINUX: error: call to undeclared function '_InterlockedAdd'

long test_InterlockedAdd_acq(int32_t volatile *Addend, long Value) {
  return _InterlockedAdd_acq(Addend, Value);
}

// CHECK-LABEL: define {{.*}} i32 @test_InterlockedAdd_acq(ptr %Addend, i32 %Value) {{.*}} {
// CHECK-MSVC: %[[OLDVAL:[0-9]+]] = atomicrmw add ptr %2, i32 %1 acquire, align 4
// CHECK-MSVC: %[[NEWVAL:[0-9]+]] = add i32 %[[OLDVAL:[0-9]+]], %1
// CHECK-MSVC: ret i32 %[[NEWVAL:[0-9]+]]
// CHECK-LINUX: error: call to undeclared function '_InterlockedAdd_acq'

long test_InterlockedAdd_nf(int32_t volatile *Addend, long Value) {
  return _InterlockedAdd_nf(Addend, Value);
}

// CHECK-LABEL: define {{.*}} i32 @test_InterlockedAdd_nf(ptr %Addend, i32 %Value) {{.*}} {
// CHECK-MSVC: %[[OLDVAL:[0-9]+]] = atomicrmw add ptr %2, i32 %1 monotonic, align 4
// CHECK-MSVC: %[[NEWVAL:[0-9]+]] = add i32 %[[OLDVAL:[0-9]+]], %1
// CHECK-MSVC: ret i32 %[[NEWVAL:[0-9]+]]
// CHECK-LINUX: error: call to undeclared function '_InterlockedAdd_nf'

long test_InterlockedAdd_rel(int32_t volatile *Addend, long Value) {
  return _InterlockedAdd_rel(Addend, Value);
}

// CHECK-LABEL: define {{.*}} i32 @test_InterlockedAdd_rel(ptr %Addend, i32 %Value) {{.*}} {
// CHECK-MSVC: %[[OLDVAL:[0-9]+]] = atomicrmw add ptr %2, i32 %1 release, align 4
// CHECK-MSVC: %[[NEWVAL:[0-9]+]] = add i32 %[[OLDVAL:[0-9]+]], %1
// CHECK-MSVC: ret i32 %[[NEWVAL:[0-9]+]]
// CHECK-LINUX: error: call to undeclared function '_InterlockedAdd_rel'

__int64 test_InterlockedAdd64(__int64 volatile *Addend, __int64 Value) {
  return _InterlockedAdd64(Addend, Value);
}

__int64 test_InterlockedAdd64_constant(__int64 volatile *Addend) {
  return _InterlockedAdd64(Addend, -1);
}

// CHECK-LABEL: define {{.*}} i64 @test_InterlockedAdd64(ptr %Addend, i64 %Value) {{.*}} {
// CHECK-MSVC: %[[OLDVAL:[0-9]+]] = atomicrmw add ptr %2, i64 %1 seq_cst, align 8
// CHECK-MSVC: %[[NEWVAL:[0-9]+]] = add i64 %[[OLDVAL:[0-9]+]], %1
// CHECK-MSVC: ret i64 %[[NEWVAL:[0-9]+]]
// CHECK-LINUX: error: call to undeclared function '_InterlockedAdd64'

__int64 test_InterlockedAdd64_acq(__int64 volatile *Addend, __int64 Value) {
  return _InterlockedAdd64_acq(Addend, Value);
}

// CHECK-LABEL: define {{.*}} i64 @test_InterlockedAdd64_acq(ptr %Addend, i64 %Value) {{.*}} {
// CHECK-MSVC: %[[OLDVAL:[0-9]+]] = atomicrmw add ptr %2, i64 %1 acquire, align 8
// CHECK-MSVC: %[[NEWVAL:[0-9]+]] = add i64 %[[OLDVAL:[0-9]+]], %1
// CHECK-MSVC: ret i64 %[[NEWVAL:[0-9]+]]
// CHECK-LINUX: error: call to undeclared function '_InterlockedAdd64_acq'

__int64 test_InterlockedAdd64_nf(__int64 volatile *Addend, __int64 Value) {
  return _InterlockedAdd64_nf(Addend, Value);
}

// CHECK-LABEL: define {{.*}} i64 @test_InterlockedAdd64_nf(ptr %Addend, i64 %Value) {{.*}} {
// CHECK-MSVC: %[[OLDVAL:[0-9]+]] = atomicrmw add ptr %2, i64 %1 monotonic, align 8
// CHECK-MSVC: %[[NEWVAL:[0-9]+]] = add i64 %[[OLDVAL:[0-9]+]], %1
// CHECK-MSVC: ret i64 %[[NEWVAL:[0-9]+]]
// CHECK-LINUX: error: call to undeclared function '_InterlockedAdd64_nf'

__int64 test_InterlockedAdd64_rel(__int64 volatile *Addend, __int64 Value) {
  return _InterlockedAdd64_rel(Addend, Value);
}

// CHECK-LABEL: define {{.*}} i64 @test_InterlockedAdd64_rel(ptr %Addend, i64 %Value) {{.*}} {
// CHECK-MSVC: %[[OLDVAL:[0-9]+]] = atomicrmw add ptr %2, i64 %1 release, align 8
// CHECK-MSVC: %[[NEWVAL:[0-9]+]] = add i64 %[[OLDVAL:[0-9]+]], %1
// CHECK-MSVC: ret i64 %[[NEWVAL:[0-9]+]]
// CHECK-LINUX: error: call to undeclared function '_InterlockedAdd64_rel'

void check_ReadWriteBarrier(void) {
  _ReadWriteBarrier();
}

// CHECK-MSVC: fence syncscope("singlethread")
// CHECK-LINUX: error: call to undeclared function '_ReadWriteBarrier'

long long check_mulh(long long a, long long b) {
  return __mulh(a, b);
}

// CHECK-MSVC: %[[ARG1:.*]] = sext i64 {{.*}} to i128
// CHECK-MSVC: %[[ARG2:.*]] = sext i64 {{.*}} to i128
// CHECK-MSVC: %[[PROD:.*]] = mul nsw i128 %[[ARG1]], %[[ARG2]]
// CHECK-MSVC: %[[HIGH:.*]] = ashr i128 %[[PROD]], 64
// CHECK-MSVC: %[[RES:.*]] = trunc i128 %[[HIGH]] to i64
// CHECK-LINUX: error: call to undeclared function '__mulh'

unsigned long long check_umulh(unsigned long long a, unsigned long long b) {
  return __umulh(a, b);
}

// CHECK-MSVC: %[[ARG1:.*]] = zext i64 {{.*}} to i128
// CHECK-MSVC: %[[ARG2:.*]] = zext i64 {{.*}} to i128
// CHECK-MSVC: %[[PROD:.*]] = mul nuw i128 %[[ARG1]], %[[ARG2]]
// CHECK-MSVC: %[[HIGH:.*]] = lshr i128 %[[PROD]], 64
// CHECK-MSVC: %[[RES:.*]] = trunc i128 %[[HIGH]] to i64
// CHECK-LINUX: error: call to undeclared function '__umulh'

void check__break() {
  __break(0);
}

// CHECK-MSVC: call void @llvm.aarch64.break(i32 0)
// CHECK-LINUX: error: call to undeclared function '__break'

void check__hlt() {
  __hlt(0);
  __hlt(1, 2, 3, 4, 5);
  int x = __hlt(0);
}

// CHECK-MSVC: call void @llvm.aarch64.hlt(i32 0)
// CHECK-LINUX: error: call to undeclared function '__hlt'

unsigned __int64 check__getReg(void) {
  unsigned volatile __int64 reg;
  reg = __getReg(18);
  reg = __getReg(31);
  return reg;
}

// CHECK-MSCOMPAT: call i64 @llvm.read_volatile_register.i64(metadata ![[MD2:.*]])
// CHECK-MSCOMPAT: call i64 @llvm.read_volatile_register.i64(metadata ![[MD3:.*]])

void test__setReg(unsigned __int64 v)
{
  __setReg(18, v);
  __setReg(31, v);
}

// CHECK-MSCOMPAT-LABEL: define{{.*}}void @test__setReg(i64{{.*}}%v){{.*}}{
// CHECK-MSCOMPAT: %[[DATA_ADDR1:.*]] = load i64, ptr %v.addr, align 8
// CHECK-MSCOMPAT: call void @llvm.write_volatile_register.i64(metadata ![[MD2]], i64 %[[DATA_ADDR1]])
// CHECK-MSCOMPAT: %[[DATA_ADDR2:.*]] = load i64, ptr %v.addr, align 8
// CHECK-MSCOMPAT: call void @llvm.write_volatile_register.i64(metadata ![[MD3]], i64 %[[DATA_ADDR2]])
// CHECK-LINUX: error: call to undeclared function '__setReg'

double test__getRegFp(void)
{
  double volatile reg;
  reg = __getRegFp(5);
  reg = __getRegFp(31);
  return reg;
}

// CHECK-MSCOMPAT-LABEL: define{{.*}}double @test__getRegFp(){{.*}}{
// CHECK-MSCOMPAT:       [[BITS:%.*]] = call i64 @llvm.read_volatile_register.i64(metadata ![[MD4:.*]])
// CHECK-MSCOMPAT:       bitcast i64 [[BITS]] to double
// CHECK-MSCOMPAT:       [[BITS:%.*]] = call i64 @llvm.read_volatile_register.i64(metadata ![[MD5:.*]])
// CHECK-MSCOMPAT:       bitcast i64 [[BITS]] to double
// CHECK-LINUX: error: call to undeclared function '__getRegFp'

void test__setRegFp(double v)
{
  __setRegFp(5, v);
  __setRegFp(31, v);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}void @test__setRegFp(double{{.*}}%v){{.*}}{
// CHECK-MSCOMPAT:       [[BITS:%.*]] = bitcast double {{.*}} to i64
// CHECK-MSCOMPAT:       call void @llvm.write_volatile_register.i64(metadata ![[MD4:.*]], i64 [[BITS]])
// CHECK-MSCOMPAT:       [[BITS:%.*]] = bitcast double {{.*}} to i64
// CHECK-MSCOMPAT:       call void @llvm.write_volatile_register.i64(metadata ![[MD5:.*]], i64 [[BITS]])
// CHECK-LINUX: error: call to undeclared function '__setRegFp'

#ifdef __LP64__
#define LONG __int32
#else
#define LONG long
#endif

#ifdef __LP64__
void check__writex18byte(unsigned char data, unsigned LONG offset) {
#else
void check__writex18byte(unsigned LONG offset, unsigned char data) {
#endif
  __writex18byte(offset, data);
}

// CHECK-MSCOMPAT: %[[DATA_ADDR:.*]] = alloca i8, align 1
// CHECK-MSCOMPAT: %[[OFFSET_ADDR:.*]] = alloca i32, align 4
// CHECK-MSCOMPAT: store i8 %data, ptr %[[DATA_ADDR]], align 1
// CHECK-MSCOMPAT: store i32 %offset, ptr %[[OFFSET_ADDR]], align 4
// CHECK-MSCOMPAT: %[[OFFSET:.*]] = load i32, ptr %[[OFFSET_ADDR]], align 4
// CHECK-MSCOMPAT: %[[DATA:.*]] = load i8, ptr %[[DATA_ADDR]], align 1
// CHECK-MSCOMPAT: %[[X18:.*]] = call i64 @llvm.read_register.i64(metadata ![[MD2]])
// CHECK-MSCOMPAT: %[[X18_AS_PTR:.*]] = inttoptr i64 %[[X18]] to ptr
// CHECK-MSCOMPAT: %[[ZEXT_OFFSET:.*]] = zext i32 %[[OFFSET]] to i64
// CHECK-MSCOMPAT: %[[PTR:.*]] = getelementptr i8, ptr %[[X18_AS_PTR]], i64 %[[ZEXT_OFFSET]]
// CHECK-MSCOMPAT: store i8 %[[DATA]], ptr %[[PTR]], align 1

#ifdef __LP64__
void check__writex18word(unsigned short data, unsigned LONG offset) {
#else
void check__writex18word(unsigned LONG offset, unsigned short data) {
#endif
  __writex18word(offset, data);
}

// CHECK-MSCOMPAT: %[[DATA_ADDR:.*]] = alloca i16, align 2
// CHECK-MSCOMPAT: %[[OFFSET_ADDR:.*]] = alloca i32, align 4
// CHECK-MSCOMPAT: store i16 %data, ptr %[[DATA_ADDR]], align 2
// CHECK-MSCOMPAT: store i32 %offset, ptr %[[OFFSET_ADDR]], align 4
// CHECK-MSCOMPAT: %[[OFFSET:.*]] = load i32, ptr %[[OFFSET_ADDR]], align 4
// CHECK-MSCOMPAT: %[[DATA:.*]] = load i16, ptr %[[DATA_ADDR]], align 2
// CHECK-MSCOMPAT: %[[X18:.*]] = call i64 @llvm.read_register.i64(metadata ![[MD2]])
// CHECK-MSCOMPAT: %[[X18_AS_PTR:.*]] = inttoptr i64 %[[X18]] to ptr
// CHECK-MSCOMPAT: %[[ZEXT_OFFSET:.*]] = zext i32 %[[OFFSET]] to i64
// CHECK-MSCOMPAT: %[[PTR:.*]] = getelementptr i8, ptr %[[X18_AS_PTR]], i64 %[[ZEXT_OFFSET]]
// CHECK-MSCOMPAT: store i16 %[[DATA]], ptr %[[PTR]], align 1

#ifdef __LP64__
void check__writex18dword(unsigned LONG data, unsigned LONG offset) {
#else
void check__writex18dword(unsigned LONG offset, unsigned LONG data) {
#endif
  __writex18dword(offset, data);
}

// CHECK-MSCOMPAT: %[[DATA_ADDR:.*]] = alloca i32, align 4
// CHECK-MSCOMPAT: %[[OFFSET_ADDR:.*]] = alloca i32, align 4
// CHECK-MSCOMPAT: store i32 %data, ptr %[[DATA_ADDR]], align 4
// CHECK-MSCOMPAT: store i32 %offset, ptr %[[OFFSET_ADDR]], align 4
// CHECK-MSCOMPAT: %[[OFFSET:.*]] = load i32, ptr %[[OFFSET_ADDR]], align 4
// CHECK-MSCOMPAT: %[[DATA:.*]] = load i32, ptr %[[DATA_ADDR]], align 4
// CHECK-MSCOMPAT: %[[X18:.*]] = call i64 @llvm.read_register.i64(metadata ![[MD2]])
// CHECK-MSCOMPAT: %[[X18_AS_PTR:.*]] = inttoptr i64 %[[X18]] to ptr
// CHECK-MSCOMPAT: %[[ZEXT_OFFSET:.*]] = zext i32 %[[OFFSET]] to i64
// CHECK-MSCOMPAT: %[[PTR:.*]] = getelementptr i8, ptr %[[X18_AS_PTR]], i64 %[[ZEXT_OFFSET]]
// CHECK-MSCOMPAT: store i32 %[[DATA]], ptr %[[PTR]], align 1

#ifdef __LP64__
void check__writex18qword(unsigned __int64 data, unsigned LONG offset) {
#else
void check__writex18qword(unsigned LONG offset, unsigned __int64 data) {
#endif
  __writex18qword(offset, data);
}

// CHECK-MSCOMPAT: %[[DATA_ADDR:.*]] = alloca i64, align 8
// CHECK-MSCOMPAT: %[[OFFSET_ADDR:.*]] = alloca i32, align 4
// CHECK-MSCOMPAT: store i64 %data, ptr %[[DATA_ADDR]], align 8
// CHECK-MSCOMPAT: store i32 %offset, ptr %[[OFFSET_ADDR]], align 4
// CHECK-MSCOMPAT: %[[OFFSET:.*]] = load i32, ptr %[[OFFSET_ADDR]], align 4
// CHECK-MSCOMPAT: %[[DATA:.*]] = load i64, ptr %[[DATA_ADDR]], align 8
// CHECK-MSCOMPAT: %[[X18:.*]] = call i64 @llvm.read_register.i64(metadata ![[MD2]])
// CHECK-MSCOMPAT: %[[X18_AS_PTR:.*]] = inttoptr i64 %[[X18]] to ptr
// CHECK-MSCOMPAT: %[[ZEXT_OFFSET:.*]] = zext i32 %[[OFFSET]] to i64
// CHECK-MSCOMPAT: %[[PTR:.*]] = getelementptr i8, ptr %[[X18_AS_PTR]], i64 %[[ZEXT_OFFSET]]
// CHECK-MSCOMPAT: store i64 %[[DATA]], ptr %[[PTR]], align 1

unsigned char check__readx18byte(unsigned LONG offset) {
  return __readx18byte(offset);
}

// CHECK-MSCOMPAT: %[[OFFSET_ADDR:.*]] = alloca i32, align 4
// CHECK-MSCOMPAT: store i32 %offset, ptr %[[OFFSET_ADDR]], align 4
// CHECK-MSCOMPAT: %[[OFFSET:.*]] = load i32, ptr %[[OFFSET_ADDR]], align 4
// CHECK-MSCOMPAT: %[[X18:.*]] = call i64 @llvm.read_register.i64(metadata ![[MD2]])
// CHECK-MSCOMPAT: %[[X18_AS_PTR:.*]] = inttoptr i64 %[[X18]] to ptr
// CHECK-MSCOMPAT: %[[ZEXT_OFFSET:.*]] = zext i32 %[[OFFSET]] to i64
// CHECK-MSCOMPAT: %[[PTR:.*]] = getelementptr i8, ptr %[[X18_AS_PTR]], i64 %[[ZEXT_OFFSET]]
// CHECK-MSCOMPAT: %[[RETVAL:.*]] = load i8, ptr %[[PTR]], align 1
// CHECK-MSCOMPAT: ret i8 %[[RETVAL]]

unsigned short check__readx18word(unsigned LONG offset) {
  return __readx18word(offset);
}

// CHECK-MSCOMPAT: %[[OFFSET_ADDR:.*]] = alloca i32, align 4
// CHECK-MSCOMPAT: store i32 %offset, ptr %[[OFFSET_ADDR]], align 4
// CHECK-MSCOMPAT: %[[OFFSET:.*]] = load i32, ptr %[[OFFSET_ADDR]], align 4
// CHECK-MSCOMPAT: %[[X18:.*]] = call i64 @llvm.read_register.i64(metadata ![[MD2]])
// CHECK-MSCOMPAT: %[[X18_AS_PTR:.*]] = inttoptr i64 %[[X18]] to ptr
// CHECK-MSCOMPAT: %[[ZEXT_OFFSET:.*]] = zext i32 %[[OFFSET]] to i64
// CHECK-MSCOMPAT: %[[PTR:.*]] = getelementptr i8, ptr %[[X18_AS_PTR]], i64 %[[ZEXT_OFFSET]]
// CHECK-MSCOMPAT: %[[RETVAL:.*]] = load i16, ptr %[[PTR]], align 1
// CHECK-MSCOMPAT: ret i16 %[[RETVAL]]

unsigned LONG check__readx18dword(unsigned LONG offset) {
  return __readx18dword(offset);
}

// CHECK-MSCOMPAT: %[[OFFSET_ADDR:.*]] = alloca i32, align 4
// CHECK-MSCOMPAT: store i32 %offset, ptr %[[OFFSET_ADDR]], align 4
// CHECK-MSCOMPAT: %[[OFFSET:.*]] = load i32, ptr %[[OFFSET_ADDR]], align 4
// CHECK-MSCOMPAT: %[[X18:.*]] = call i64 @llvm.read_register.i64(metadata ![[MD2]])
// CHECK-MSCOMPAT: %[[X18_AS_PTR:.*]] = inttoptr i64 %[[X18]] to ptr
// CHECK-MSCOMPAT: %[[ZEXT_OFFSET:.*]] = zext i32 %[[OFFSET]] to i64
// CHECK-MSCOMPAT: %[[PTR:.*]] = getelementptr i8, ptr %[[X18_AS_PTR]], i64 %[[ZEXT_OFFSET]]
// CHECK-MSCOMPAT: %[[RETVAL:.*]] = load i32, ptr %[[PTR]], align 1
// CHECK-MSCOMPAT: ret i32 %[[RETVAL]]

unsigned __int64 check__readx18qword(unsigned LONG offset) {
  return __readx18qword(offset);
}

// CHECK-MSCOMPAT: %[[OFFSET_ADDR:.*]] = alloca i32, align 4
// CHECK-MSCOMPAT: store i32 %offset, ptr %[[OFFSET_ADDR]], align 4
// CHECK-MSCOMPAT: %[[OFFSET:.*]] = load i32, ptr %[[OFFSET_ADDR]], align 4
// CHECK-MSCOMPAT: %[[X18:.*]] = call i64 @llvm.read_register.i64(metadata ![[MD2]])
// CHECK-MSCOMPAT: %[[X18_AS_PTR:.*]] = inttoptr i64 %[[X18]] to ptr
// CHECK-MSCOMPAT: %[[ZEXT_OFFSET:.*]] = zext i32 %[[OFFSET]] to i64
// CHECK-MSCOMPAT: %[[PTR:.*]] = getelementptr i8, ptr %[[X18_AS_PTR]], i64 %[[ZEXT_OFFSET]]
// CHECK-MSCOMPAT: %[[RETVAL:.*]] = load i64, ptr %[[PTR]], align 1
// CHECK-MSCOMPAT: ret i64 %[[RETVAL]]

#ifdef __LP64__
void check__addx18byte(unsigned char data, unsigned LONG offset) {
#else
void check__addx18byte(unsigned LONG offset, unsigned char data) {
#endif
  __addx18byte(offset, data);
}

// CHECK-MSCOMPAT: %[[DATA_ADDR:.*]] = alloca i8, align 1
// CHECK-MSCOMPAT: %[[OFFSET_ADDR:.*]] = alloca i32, align 4
// CHECK-MSCOMPAT: store i8 %data, ptr %[[DATA_ADDR]], align 1
// CHECK-MSCOMPAT: store i32 %offset, ptr %[[OFFSET_ADDR]], align 4
// CHECK-MSCOMPAT: %[[OFFSET:.*]] = load i32, ptr %[[OFFSET_ADDR]], align 4
// CHECK-MSCOMPAT: %[[DATA:.*]] = load i8, ptr %[[DATA_ADDR]], align 1
// CHECK-MSCOMPAT: %[[X18:.*]] = call i64 @llvm.read_register.i64(metadata ![[MD2]])
// CHECK-MSCOMPAT: %[[X18_AS_PTR:.*]] = inttoptr i64 %[[X18]] to ptr
// CHECK-MSCOMPAT: %[[ZEXT_OFFSET:.*]] = zext i32 %[[OFFSET]] to i64
// CHECK-MSCOMPAT: %[[PTR:.*]] = getelementptr i8, ptr %[[X18_AS_PTR]], i64 %[[ZEXT_OFFSET]]
// CHECK-MSCOMPAT: %[[ORIG_VAL:.*]] = load i8, ptr %[[PTR]], align 1
// CHECK-MSCOMPAT: %[[SUM:.*]] = add i8 %[[ORIG_VAL]], %[[DATA]]
// CHECK-MSCOMPAT: store i8 %[[SUM]], ptr %[[PTR]], align 1

#ifdef __LP64__
void check__addx18word(unsigned short data, unsigned LONG offset) {
#else
void check__addx18word(unsigned LONG offset, unsigned short data) {
#endif
  __addx18word(offset, data);
}

// CHECK-MSCOMPAT: %[[DATA_ADDR:.*]] = alloca i16, align 2
// CHECK-MSCOMPAT: %[[OFFSET_ADDR:.*]] = alloca i32, align 4
// CHECK-MSCOMPAT: store i16 %data, ptr %[[DATA_ADDR]], align 2
// CHECK-MSCOMPAT: store i32 %offset, ptr %[[OFFSET_ADDR]], align 4
// CHECK-MSCOMPAT: %[[OFFSET:.*]] = load i32, ptr %[[OFFSET_ADDR]], align 4
// CHECK-MSCOMPAT: %[[DATA:.*]] = load i16, ptr %[[DATA_ADDR]], align 2
// CHECK-MSCOMPAT: %[[X18:.*]] = call i64 @llvm.read_register.i64(metadata ![[MD2]])
// CHECK-MSCOMPAT: %[[X18_AS_PTR:.*]] = inttoptr i64 %[[X18]] to ptr
// CHECK-MSCOMPAT: %[[ZEXT_OFFSET:.*]] = zext i32 %[[OFFSET]] to i64
// CHECK-MSCOMPAT: %[[PTR:.*]] = getelementptr i8, ptr %[[X18_AS_PTR]], i64 %[[ZEXT_OFFSET]]
// CHECK-MSCOMPAT: %[[ORIG_VAL:.*]] = load i16, ptr %[[PTR]], align 1
// CHECK-MSCOMPAT: %[[SUM:.*]] = add i16 %[[ORIG_VAL]], %[[DATA]]
// CHECK-MSCOMPAT: store i16 %[[SUM]], ptr %[[PTR]], align 1

#ifdef __LP64__
void check__addx18dword(unsigned LONG data, unsigned LONG offset) {
#else
void check__addx18dword(unsigned LONG offset, unsigned LONG data) {
#endif
  __addx18dword(offset, data);
}

// CHECK-MSCOMPAT: %[[DATA_ADDR:.*]] = alloca i32, align 4
// CHECK-MSCOMPAT: %[[OFFSET_ADDR:.*]] = alloca i32, align 4
// CHECK-MSCOMPAT: store i32 %data, ptr %[[DATA_ADDR]], align 4
// CHECK-MSCOMPAT: store i32 %offset, ptr %[[OFFSET_ADDR]], align 4
// CHECK-MSCOMPAT: %[[OFFSET:.*]] = load i32, ptr %[[OFFSET_ADDR]], align 4
// CHECK-MSCOMPAT: %[[DATA:.*]] = load i32, ptr %[[DATA_ADDR]], align 4
// CHECK-MSCOMPAT: %[[X18:.*]] = call i64 @llvm.read_register.i64(metadata ![[MD2]])
// CHECK-MSCOMPAT: %[[X18_AS_PTR:.*]] = inttoptr i64 %[[X18]] to ptr
// CHECK-MSCOMPAT: %[[ZEXT_OFFSET:.*]] = zext i32 %[[OFFSET]] to i64
// CHECK-MSCOMPAT: %[[PTR:.*]] = getelementptr i8, ptr %[[X18_AS_PTR]], i64 %[[ZEXT_OFFSET]]
// CHECK-MSCOMPAT: %[[ORIG_VAL:.*]] = load i32, ptr %[[PTR]], align 1
// CHECK-MSCOMPAT: %[[SUM:.*]] = add i32 %[[ORIG_VAL]], %[[DATA]]
// CHECK-MSCOMPAT: store i32 %[[SUM]], ptr %[[PTR]], align 1

#ifdef __LP64__
void check__addx18qword(unsigned __int64 data, unsigned LONG offset) {
#else
void check__addx18qword(unsigned LONG offset, unsigned __int64 data) {
#endif
  __addx18qword(offset, data);
}

// CHECK-MSCOMPAT: %[[DATA_ADDR:.*]] = alloca i64, align 8
// CHECK-MSCOMPAT: %[[OFFSET_ADDR:.*]] = alloca i32, align 4
// CHECK-MSCOMPAT: store i64 %data, ptr %[[DATA_ADDR]], align 8
// CHECK-MSCOMPAT: store i32 %offset, ptr %[[OFFSET_ADDR]], align 4
// CHECK-MSCOMPAT: %[[OFFSET:.*]] = load i32, ptr %[[OFFSET_ADDR]], align 4
// CHECK-MSCOMPAT: %[[DATA:.*]] = load i64, ptr %[[DATA_ADDR]], align 8
// CHECK-MSCOMPAT: %[[X18:.*]] = call i64 @llvm.read_register.i64(metadata ![[MD2]])
// CHECK-MSCOMPAT: %[[X18_AS_PTR:.*]] = inttoptr i64 %[[X18]] to ptr
// CHECK-MSCOMPAT: %[[ZEXT_OFFSET:.*]] = zext i32 %[[OFFSET]] to i64
// CHECK-MSCOMPAT: %[[PTR:.*]] = getelementptr i8, ptr %[[X18_AS_PTR]], i64 %[[ZEXT_OFFSET]]
// CHECK-MSCOMPAT: %[[ORIG_VAL:.*]] = load i64, ptr %[[PTR]], align 1
// CHECK-MSCOMPAT: %[[SUM:.*]] = add i64 %[[ORIG_VAL]], %[[DATA]]
// CHECK-MSCOMPAT: store i64 %[[SUM]], ptr %[[PTR]], align 1

void check__incx18byte(unsigned LONG offset) {
  __incx18byte(offset);
}

// CHECK-MSCOMPAT: %[[OFFSET_ADDR:.*]] = alloca i32, align 4
// CHECK-MSCOMPAT: store i32 %offset, ptr %[[OFFSET_ADDR]], align 4
// CHECK-MSCOMPAT: %[[OFFSET:.*]] = load i32, ptr %[[OFFSET_ADDR]], align 4
// CHECK-MSCOMPAT: %[[X18:.*]] = call i64 @llvm.read_register.i64(metadata ![[MD2]])
// CHECK-MSCOMPAT: %[[X18_AS_PTR:.*]] = inttoptr i64 %[[X18]] to ptr
// CHECK-MSCOMPAT: %[[ZEXT_OFFSET:.*]] = zext i32 %[[OFFSET]] to i64
// CHECK-MSCOMPAT: %[[PTR:.*]] = getelementptr i8, ptr %[[X18_AS_PTR]], i64 %[[ZEXT_OFFSET]]
// CHECK-MSCOMPAT: %[[ORIG_VAL:.*]] = load i8, ptr %[[PTR]], align 1
// CHECK-MSCOMPAT: %[[SUM:.*]] = add i8 %[[ORIG_VAL]], 1
// CHECK-MSCOMPAT: store i8 %[[SUM]], ptr %[[PTR]], align 1

void check__incx18word(unsigned LONG offset) {
  __incx18word(offset);
}

// CHECK-MSCOMPAT: %[[OFFSET_ADDR:.*]] = alloca i32, align 4
// CHECK-MSCOMPAT: store i32 %offset, ptr %[[OFFSET_ADDR]], align 4
// CHECK-MSCOMPAT: %[[OFFSET:.*]] = load i32, ptr %[[OFFSET_ADDR]], align 4
// CHECK-MSCOMPAT: %[[X18:.*]] = call i64 @llvm.read_register.i64(metadata ![[MD2]])
// CHECK-MSCOMPAT: %[[X18_AS_PTR:.*]] = inttoptr i64 %[[X18]] to ptr
// CHECK-MSCOMPAT: %[[ZEXT_OFFSET:.*]] = zext i32 %[[OFFSET]] to i64
// CHECK-MSCOMPAT: %[[PTR:.*]] = getelementptr i8, ptr %[[X18_AS_PTR]], i64 %[[ZEXT_OFFSET]]
// CHECK-MSCOMPAT: %[[ORIG_VAL:.*]] = load i16, ptr %[[PTR]], align 1
// CHECK-MSCOMPAT: %[[SUM:.*]] = add i16 %[[ORIG_VAL]], 1
// CHECK-MSCOMPAT: store i16 %[[SUM]], ptr %[[PTR]], align 1

void check__incx18dword(unsigned LONG offset) {
  __incx18dword(offset);
}

// CHECK-MSCOMPAT: %[[OFFSET_ADDR:.*]] = alloca i32, align 4
// CHECK-MSCOMPAT: store i32 %offset, ptr %[[OFFSET_ADDR]], align 4
// CHECK-MSCOMPAT: %[[OFFSET:.*]] = load i32, ptr %[[OFFSET_ADDR]], align 4
// CHECK-MSCOMPAT: %[[X18:.*]] = call i64 @llvm.read_register.i64(metadata ![[MD2]])
// CHECK-MSCOMPAT: %[[X18_AS_PTR:.*]] = inttoptr i64 %[[X18]] to ptr
// CHECK-MSCOMPAT: %[[ZEXT_OFFSET:.*]] = zext i32 %[[OFFSET]] to i64
// CHECK-MSCOMPAT: %[[PTR:.*]] = getelementptr i8, ptr %[[X18_AS_PTR]], i64 %[[ZEXT_OFFSET]]
// CHECK-MSCOMPAT: %[[ORIG_VAL:.*]] = load i32, ptr %[[PTR]], align 1
// CHECK-MSCOMPAT: %[[SUM:.*]] = add i32 %[[ORIG_VAL]], 1
// CHECK-MSCOMPAT: store i32 %[[SUM]], ptr %[[PTR]], align 1

void check__incx18qword(unsigned LONG offset) {
  __incx18qword(offset);
}

// CHECK-MSCOMPAT: %[[OFFSET_ADDR:.*]] = alloca i32, align 4
// CHECK-MSCOMPAT: store i32 %offset, ptr %[[OFFSET_ADDR]], align 4
// CHECK-MSCOMPAT: %[[OFFSET:.*]] = load i32, ptr %[[OFFSET_ADDR]], align 4
// CHECK-MSCOMPAT: %[[X18:.*]] = call i64 @llvm.read_register.i64(metadata ![[MD2]])
// CHECK-MSCOMPAT: %[[X18_AS_PTR:.*]] = inttoptr i64 %[[X18]] to ptr
// CHECK-MSCOMPAT: %[[ZEXT_OFFSET:.*]] = zext i32 %[[OFFSET]] to i64
// CHECK-MSCOMPAT: %[[PTR:.*]] = getelementptr i8, ptr %[[X18_AS_PTR]], i64 %[[ZEXT_OFFSET]]
// CHECK-MSCOMPAT: %[[ORIG_VAL:.*]] = load i64, ptr %[[PTR]], align 1
// CHECK-MSCOMPAT: %[[SUM:.*]] = add i64 %[[ORIG_VAL]], 1
// CHECK-MSCOMPAT: store i64 %[[SUM]], ptr %[[PTR]], align 1

double check__CopyDoubleFromInt64(__int64 arg1) {
  return _CopyDoubleFromInt64(arg1);
}

// CHECK-MSCOMPAT: %[[ARG:.*]].addr = alloca i64, align 8
// CHECK-MSCOMPAT: store i64 %[[ARG]], ptr %[[ARG]].addr, align 8
// CHECK-MSCOMPAT: %[[VAR0:.*]] = load i64, ptr %[[ARG]].addr, align 8
// CHECK-MSCOMPAT: %[[VAR1:.*]] = bitcast i64 %[[VAR0]] to double
// CHECK-MSCOMPAT: ret double %[[VAR1]]
// CHECK-LINUX: error: call to undeclared function '_CopyDoubleFromInt64'

float check__CopyFloatFromInt32(__int32 arg1) {
  return _CopyFloatFromInt32(arg1);
}

// CHECK-MSCOMPAT: %[[ARG:.*]].addr = alloca i32, align 4
// CHECK-MSCOMPAT: store i32 %[[ARG]], ptr %[[ARG]].addr, align 4
// CHECK-MSCOMPAT: %[[VAR0:.*]] = load i32, ptr %[[ARG]].addr, align 4
// CHECK-MSCOMPAT: %[[VAR1:.*]] = bitcast i32 %[[VAR0]] to float
// CHECK-MSCOMPAT: ret float %[[VAR1]]
// CHECK-LINUX: error: call to undeclared function '_CopyFloatFromInt32'

__int32 check__CopyInt32FromFloat(float arg1) {
  return _CopyInt32FromFloat(arg1);
}

// CHECK-MSCOMPAT: %[[ARG:.*]].addr = alloca float, align 4
// CHECK-MSCOMPAT: store float %[[ARG]], ptr %[[ARG]].addr, align 4
// CHECK-MSCOMPAT: %[[VAR0:.*]] = load float, ptr %[[ARG]].addr, align 4
// CHECK-MSCOMPAT: %[[VAR1:.*]] = bitcast float %[[VAR0]] to i32
// CHECK-MSCOMPAT: ret i32 %[[VAR1]]
// CHECK-LINUX: error: call to undeclared function '_CopyInt32FromFloat'

__int64 check__CopyInt64FromDouble(double arg1) {
  return _CopyInt64FromDouble(arg1);
}

// CHECK-MSCOMPAT: %[[ARG:.*]].addr = alloca double, align 8
// CHECK-MSCOMPAT: store double %[[ARG]], ptr %[[ARG]].addr, align 8
// CHECK-MSCOMPAT: %[[VAR0:.*]] = load double, ptr %[[ARG]].addr, align 8
// CHECK-MSCOMPAT: %[[VAR1:.*]] = bitcast double %[[VAR0]] to i64
// CHECK-MSCOMPAT: ret i64 %[[VAR1]]
// CHECK-LINUX: error: call to undeclared function '_CopyInt64FromDouble'

unsigned int check__CountLeadingOnes(unsigned LONG arg1) {
  return _CountLeadingOnes(arg1);
}

// CHECK-MSCOMPAT: %[[ARG1:.*]].addr = alloca i32, align 4
// CHECK-MSCOMPAT: store i32 %[[ARG1]], ptr %[[ARG1]].addr, align 4
// CHECK-MSCOMPAT: %[[VAR0:.*]] = load i32, ptr %[[ARG1]].addr, align 4
// CHECK-MSCOMPAT: %[[VAR1:.*]] = xor i32 %[[VAR0]], -1
// CHECK-MSCOMPAT: %[[VAR2:.*]] = call i32 @llvm.ctlz.i32(i32 %1, i1 false)
// CHECK-MSCOMPAT: ret i32 %[[VAR2]]
// CHECK-LINUX: error: call to undeclared function '_CountLeadingOnes'

unsigned int check__CountLeadingOnes64(unsigned __int64 arg1) {
  return _CountLeadingOnes64(arg1);
}

// CHECK-MSCOMPAT: %[[ARG1:.*]].addr = alloca i64, align 8
// CHECK-MSCOMPAT: store i64 %[[ARG1]], ptr %[[ARG1]].addr, align 8
// CHECK-MSCOMPAT: %[[VAR0:.*]] = load i64, ptr %[[ARG1]].addr, align 8
// CHECK-MSCOMPAT: %[[VAR1:.*]] = xor i64 %[[VAR0]], -1
// CHECK-MSCOMPAT: %[[VAR2:.*]] = call i64 @llvm.ctlz.i64(i64 %1, i1 false)
// CHECK-MSCOMPAT: %[[VAR3:.*]] = trunc i64 %2 to i32
// CHECK-MSCOMPAT: ret i32 %[[VAR3]]
// CHECK-LINUX: error: call to undeclared function '_CountLeadingOnes64'

unsigned int check__CountLeadingSigns(__int32 arg1) {
  return _CountLeadingSigns(arg1);
}

// CHECK-MSCOMPAT: %[[ARG1:.*]].addr = alloca i32, align 4
// CHECK-MSCOMPAT: store i32 %[[ARG1]], ptr %[[ARG1]].addr, align 4
// CHECK-MSCOMPAT: %[[VAR0:.*]] = load i32, ptr %[[ARG1]].addr, align 4
// CHECK-MSCOMPAT: %[[CLS:.*]] = call i32 @llvm.aarch64.cls(i32 %[[VAR0]])
// CHECK-MSCOMPAT: ret i32 %[[CLS]]
// CHECK-LINUX: error: call to undeclared function '_CountLeadingSigns'

unsigned int check__CountLeadingSigns64(__int64 arg1) {
  return _CountLeadingSigns64(arg1);
}

// CHECK-MSCOMPAT: %[[ARG1:.*]].addr = alloca i64, align 8
// CHECK-MSCOMPAT: store i64 %[[ARG1]], ptr %[[ARG1]].addr, align 8
// CHECK-MSCOMPAT: %[[VAR0:.*]] = load i64, ptr %[[ARG1]].addr, align 8
// CHECK-MSCOMPAT: %[[CLS:.*]] = call i32 @llvm.aarch64.cls64(i64 %[[VAR0]])
// CHECK-MSCOMPAT: ret i32 %[[CLS]]
// CHECK-LINUX: error: call to undeclared function '_CountLeadingSigns64'

unsigned int check__CountLeadingZeros(__int32 arg1) {
  return _CountLeadingZeros(arg1);
}

// CHECK-MSCOMPAT: %[[ARG1:.*]].addr = alloca i32, align 4
// CHECK-MSCOMPAT: store i32 %[[ARG1]], ptr %[[ARG1]].addr, align 4
// CHECK-MSCOMPAT: %[[VAR0:.*]] = load i32, ptr %[[ARG1]].addr, align 4
// CHECK-MSCOMPAT: %[[VAR1:.*]] = call i32 @llvm.ctlz.i32(i32 %[[VAR0]], i1 false)
// CHECK-MSCOMPAT: ret i32 %[[VAR1]]
// CHECK-LINUX: error: call to undeclared function '_CountLeadingZeros'

unsigned int check__CountLeadingZeros64(__int64 arg1) {
  return _CountLeadingZeros64(arg1);
}

// CHECK-MSCOMPAT: %[[ARG1:.*]].addr = alloca i64, align 8
// CHECK-MSCOMPAT: store i64 %[[ARG1]], ptr %[[ARG1]].addr, align 8
// CHECK-MSCOMPAT: %[[VAR0:.*]] = load i64, ptr %[[ARG1]].addr, align 8
// CHECK-MSCOMPAT: %[[VAR1:.*]] = call i64 @llvm.ctlz.i64(i64 %[[VAR0]], i1 false)
// CHECK-MSCOMPAT: %[[VAR2:.*]] = trunc i64 %[[VAR1]] to i32
// CHECK-MSCOMPAT: ret i32 %[[VAR2]]
// CHECK-LINUX: error: call to undeclared function '_CountLeadingZeros64'

unsigned int check_CountOneBits(unsigned LONG arg1) {
  return _CountOneBits(arg1);
}

// CHECK-MSCOMPAT: %[[ARG1:.*]].addr = alloca i32, align 4
// CHECK-MSCOMPAT: store i32 %[[ARG1]], ptr %[[ARG1]].addr, align 4
// CHECK-MSCOMPAT: %[[VAR0:.*]] = load i32, ptr %[[ARG1]].addr, align 4
// CHECK-MSCOMPAT: %[[VAR1:.*]] = call i32 @llvm.ctpop.i32(i32 %0)
// CHECK-MSCOMPAT: ret i32 %[[VAR1]]
// CHECK-LINUX: error: call to undeclared function '_CountOneBits'

unsigned int check_CountOneBits64(unsigned __int64 arg1) {
  return _CountOneBits64(arg1);
}

// CHECK-MSCOMPAT: %[[ARG1:.*]].addr = alloca i64, align 8
// CHECK-MSCOMPAT: store i64 %[[ARG1]], ptr %[[ARG1]].addr, align 8
// CHECK-MSCOMPAT: %[[VAR0:.*]] = load i64, ptr %[[ARG1]].addr, align 8
// CHECK-MSCOMPAT: %[[VAR1:.*]] = call i64 @llvm.ctpop.i64(i64 %0)
// CHECK-MSCOMPAT: %[[VAR2:.*]] = trunc i64 %1 to i32
// CHECK-MSCOMPAT: ret i32 %[[VAR2]]
// CHECK-LINUX: error: call to undeclared function '_CountOneBits64'

unsigned int check_CountTrailingZeros(unsigned LONG arg1) {
  return _CountTrailingZeros(arg1);
}

// CHECK-MSCOMPAT: %[[ARG1:.*]].addr = alloca i32, align 4
// CHECK-MSCOMPAT: store i32 %[[ARG1]], ptr %[[ARG1]].addr, align 4
// CHECK-MSCOMPAT: %[[VAR0:.*]] = load i32, ptr %[[ARG1]].addr, align 4
// CHECK-MSCOMPAT: %[[VAR1:.*]] = call i32 @llvm.cttz.i32(i32 %[[VAR0]], i1 false)
// CHECK-MSCOMPAT: ret i32 %[[VAR1]]
// CHECK-LINUX: error: call to undeclared function '_CountTrailingZeros'

unsigned int check_CountTrailingZeros64(unsigned __int64 arg1) {
  return _CountTrailingZeros64(arg1);
}

// CHECK-MSCOMPAT: %[[ARG1:.*]].addr = alloca i64, align 8
// CHECK-MSCOMPAT: store i64 %[[ARG1]], ptr %[[ARG1]].addr, align 8
// CHECK-MSCOMPAT: %[[VAR0:.*]] = load i64, ptr %[[ARG1]].addr, align 8
// CHECK-MSCOMPAT: %[[VAR1:.*]] = call i64 @llvm.cttz.i64(i64 %[[VAR0]], i1 false)
// CHECK-MSCOMPAT: %[[VAR2:.*]] = trunc i64 %[[VAR1]] to i32
// CHECK-MSCOMPAT: ret i32 %[[VAR2]]
// CHECK-LINUX: error: call to undeclared function '_CountTrailingZeros64'

void check__prefetch(void *arg1) {
  return __prefetch(arg1);
}

// CHECK-MSCOMPAT: %[[ARG1:.*]].addr = alloca ptr, align 8
// CHECK-MSCOMPAT: store ptr %[[ARG1]], ptr %[[ARG1]].addr, align 8
// CHECK-MSCOMPAT: %[[VAR0:.*]] = load ptr, ptr %[[ARG1]].addr, align 8
// CHECK-MSCOMPAT: call void @llvm.prefetch.p0(ptr %[[VAR0]], i32 0, i32 3, i32 1)
// CHECK-MSCOMPAT: ret void

void check__prefetch2(void *arg1) {
  __prefetch2(arg1, 0x00);
  __prefetch2(arg1, 0x13);
}

// CHECK-MSCOMPAT-LABEL: define{{.*}}void @check__prefetch2(ptr{{.*}}%arg1){{.*}}{
// CHECK-MSCOMPAT: call void @llvm.aarch64.prefetch(ptr %{{.*}}, i32 0, i32 0, i32 0, i32 1)
// CHECK-MSCOMPAT: call void @llvm.aarch64.prefetch(ptr %{{.*}}, i32 1, i32 1, i32 1, i32 1)
// CHECK-LINUX: error: call to undeclared function '__prefetch2'


unsigned char check__ldar8(unsigned char volatile *p) {
  return __ldar8(p);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i8 @check__ldar8(ptr{{.*}}%p){{.*}}{
// CHECK-MSCOMPAT:       %[[RET:[0-9]+]] = load atomic volatile i8, ptr %{{.*}} seq_cst, align 1
// CHECK-MSCOMPAT:       ret i8 %[[RET]]
// CHECK-LINUX: error: call to undeclared function '__ldar8'

unsigned short check__ldar16(unsigned short volatile *p) {
  return __ldar16(p);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i16 @check__ldar16(ptr{{.*}}%p){{.*}}{
// CHECK-MSCOMPAT:       %[[RET:[0-9]+]] = load atomic volatile i16, ptr %{{.*}} seq_cst, align 2
// CHECK-MSCOMPAT:       ret i16 %[[RET]]
// CHECK-LINUX: error: call to undeclared function '__ldar16'

unsigned int check__ldar32(unsigned int volatile *p) {
  return __ldar32(p);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i32 @check__ldar32(ptr{{.*}}%p){{.*}}{
// CHECK-MSCOMPAT:       %[[RET:[0-9]+]] = load atomic volatile i32, ptr %{{.*}} seq_cst, align 4
// CHECK-MSCOMPAT:       ret i32 %[[RET]]
// CHECK-LINUX: error: call to undeclared function '__ldar32'

unsigned long long int  check__ldar64(unsigned long long int volatile *p) {
  return __ldar64(p);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i64 @check__ldar64(ptr{{.*}}%p){{.*}}{
// CHECK-MSCOMPAT:       %[[RET:[0-9]+]] = load atomic volatile i64, ptr %{{.*}} seq_cst, align 8
// CHECK-MSCOMPAT:       ret i64 %[[RET]]
// CHECK-LINUX: error: call to undeclared function '__ldar64'

unsigned char check__ldxr8(unsigned char volatile *p) {
  return __ldxr8(p);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i8 @check__ldxr8(ptr{{.*}}%p){{.*}}{
// CHECK-MSCOMPAT:       %[[RET:.*]] = call i64 @llvm.aarch64.ldxr.p0(ptr elementtype(i8) %{{.*}})
// CHECK-MSCOMPAT:       trunc i64 %[[RET]] to i8
// CHECK-LINUX: error: call to undeclared function '__ldxr8'

unsigned short check__ldxr16(unsigned short volatile *p) {
  return __ldxr16(p);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i16 @check__ldxr16(ptr{{.*}}%p){{.*}}{
// CHECK-MSCOMPAT:       %[[RET:.*]] = call i64 @llvm.aarch64.ldxr.p0(ptr elementtype(i16) %{{.*}})
// CHECK-MSCOMPAT:       trunc i64 %[[RET]] to i16
// CHECK-LINUX: error: call to undeclared function '__ldxr16'

unsigned int check__ldxr32(unsigned int volatile *p) {
  return __ldxr32(p);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i32 @check__ldxr32(ptr{{.*}}%p){{.*}}{
// CHECK-MSCOMPAT:       %[[RET:.*]] = call i64 @llvm.aarch64.ldxr.p0(ptr elementtype(i32) %{{.*}})
// CHECK-MSCOMPAT:       trunc i64 %[[RET]] to i32
// CHECK-LINUX: error: call to undeclared function '__ldxr32'

unsigned long long int check__ldxr64(unsigned long long int volatile *p) {
  return __ldxr64(p);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i64 @check__ldxr64(ptr{{.*}}%p){{.*}}{
// CHECK-MSCOMPAT:       %[[RET:.*]] = call i64 @llvm.aarch64.ldxr.p0(ptr elementtype(i64) %{{.*}})
// CHECK-LINUX: error: call to undeclared function '__ldxr64'

unsigned char check__ldaxr8(unsigned char volatile *p) {
  return __ldaxr8(p);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i8 @check__ldaxr8(ptr{{.*}}%p){{.*}}{
// CHECK-MSCOMPAT:       %[[RET:.*]] = call i64 @llvm.aarch64.ldaxr.p0(ptr elementtype(i8) %{{.*}})
// CHECK-MSCOMPAT:       trunc i64 %[[RET]] to i8
// CHECK-LINUX: error: call to undeclared function '__ldaxr8'

unsigned short check__ldaxr16(unsigned short volatile *p) {
  return __ldaxr16(p);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i16 @check__ldaxr16(ptr{{.*}}%p){{.*}}{
// CHECK-MSCOMPAT:       %[[RET:.*]] = call i64 @llvm.aarch64.ldaxr.p0(ptr elementtype(i16) %{{.*}})
// CHECK-MSCOMPAT:       trunc i64 %[[RET]] to i16
// CHECK-LINUX: error: call to undeclared function '__ldaxr16'

unsigned int check__ldaxr32(unsigned int volatile *p) {
  return __ldaxr32(p);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i32 @check__ldaxr32(ptr{{.*}}%p){{.*}}{
// CHECK-MSCOMPAT:       %[[RET:.*]] = call i64 @llvm.aarch64.ldaxr.p0(ptr elementtype(i32) %{{.*}})
// CHECK-MSCOMPAT:       trunc i64 %[[RET]] to i32
// CHECK-LINUX: error: call to undeclared function '__ldaxr32'

unsigned long long int check__ldaxr64(unsigned long long int volatile *p) {
  return __ldaxr64(p);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i64 @check__ldaxr64(ptr{{.*}}%p){{.*}}{
// CHECK-MSCOMPAT:       %[[RET:.*]] = call i64 @llvm.aarch64.ldaxr.p0(ptr elementtype(i64) %{{.*}})
// CHECK-LINUX: error: call to undeclared function '__ldaxr64'

unsigned char check__stxr8(unsigned char volatile *p, unsigned char v) {
  return __stxr8(p, v);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i8 @check__stxr8(ptr{{.*}}%p, i8{{.*}}%v){{.*}}{
// CHECK-MSCOMPAT:       %[[ST:.*]] = call i32 @llvm.aarch64.stxr.p0(i64 %{{.*}}, ptr elementtype(i8) %{{.*}})
// CHECK-MSCOMPAT:       trunc i32 %[[ST]] to i8
// CHECK-LINUX: error: call to undeclared function '__stxr8'

unsigned char check__stxr16(unsigned short volatile *p, unsigned short v) {
  return __stxr16(p, v);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i8 @check__stxr16(ptr{{.*}}%p, i16{{.*}}%v){{.*}}{
// CHECK-MSCOMPAT:       %[[ST:.*]] = call i32 @llvm.aarch64.stxr.p0(i64 %{{.*}}, ptr elementtype(i16) %{{.*}})
// CHECK-MSCOMPAT:       trunc i32 %[[ST]] to i8
// CHECK-LINUX: error: call to undeclared function '__stxr16'

unsigned char check__stxr32(unsigned int volatile *p, unsigned int v) {
  return __stxr32(p, v);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i8 @check__stxr32(ptr{{.*}}%p, i32{{.*}}%v){{.*}}{
// CHECK-MSCOMPAT:       %[[ST:.*]] = call i32 @llvm.aarch64.stxr.p0(i64 %{{.*}}, ptr elementtype(i32) %{{.*}})
// CHECK-MSCOMPAT:       trunc i32 %[[ST]] to i8
// CHECK-LINUX: error: call to undeclared function '__stxr32'

unsigned char check__stxr64(unsigned long long int volatile *p, unsigned long long int v) {
  return __stxr64(p, v);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i8 @check__stxr64(ptr{{.*}}%p, i64{{.*}}%v){{.*}}{
// CHECK-MSCOMPAT:       %[[ST:.*]] = call i32 @llvm.aarch64.stxr.p0(i64 %{{.*}}, ptr elementtype(i64) %{{.*}})
// CHECK-MSCOMPAT:       trunc i32 %[[ST]] to i8
// CHECK-LINUX: error: call to undeclared function '__stxr64'

unsigned char check__stlxr8(unsigned char volatile *p, unsigned char v) {
  return __stlxr8(p, v);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i8 @check__stlxr8(ptr{{.*}}%p, i8{{.*}}%v){{.*}}{
// CHECK-MSCOMPAT:       %[[ST:.*]] = call i32 @llvm.aarch64.stlxr.p0(i64 %{{.*}}, ptr elementtype(i8) %{{.*}})
// CHECK-MSCOMPAT:       trunc i32 %[[ST]] to i8
// CHECK-LINUX: error: call to undeclared function '__stlxr8'

unsigned char check__stlxr16(unsigned short volatile *p, unsigned short v) {
  return __stlxr16(p, v);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i8 @check__stlxr16(ptr{{.*}}%p, i16{{.*}}%v){{.*}}{
// CHECK-MSCOMPAT:       %[[ST:.*]] = call i32 @llvm.aarch64.stlxr.p0(i64 %{{.*}}, ptr elementtype(i16) %{{.*}})
// CHECK-MSCOMPAT:       trunc i32 %[[ST]] to i8
// CHECK-LINUX: error: call to undeclared function '__stlxr16'

unsigned char check__stlxr32(unsigned int volatile *p, unsigned int v) {
  return __stlxr32(p, v);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i8 @check__stlxr32(ptr{{.*}}%p, i32{{.*}}%v){{.*}}{
// CHECK-MSCOMPAT:       %[[ST:.*]] = call i32 @llvm.aarch64.stlxr.p0(i64 %{{.*}}, ptr elementtype(i32) %{{.*}})
// CHECK-MSCOMPAT:       trunc i32 %[[ST]] to i8
// CHECK-LINUX: error: call to undeclared function '__stlxr32'

unsigned char check__stlxr64(unsigned long long int volatile *p, unsigned long long int v) {
  return __stlxr64(p, v);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i8 @check__stlxr64(ptr{{.*}}%p, i64{{.*}}%v){{.*}}{
// CHECK-MSCOMPAT:       %[[ST:.*]] = call i32 @llvm.aarch64.stlxr.p0(i64 %{{.*}}, ptr elementtype(i64) %{{.*}})
// CHECK-MSCOMPAT:       trunc i32 %[[ST]] to i8
// CHECK-LINUX: error: call to undeclared function '__stlxr64'

void check__clrex(void) {
  __clrex(15);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}void @check__clrex(){{.*}}{
// CHECK-MSCOMPAT:       call void @llvm.aarch64.clrex(i32 15)
// CHECK-LINUX: error: call to undeclared function '__clrex'

void test__stlr8(unsigned __int8 volatile *p, unsigned __int8 v)
{
  __stlr8 (p, v);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}void @test__stlr8(ptr{{.*}}%p, i8{{.*}}%v){{.*}}{
// CHECK-MSCOMPAT:       %[[DEST:[0-9]+]] = load ptr, ptr %p.addr, align 8
// CHECK-MSCOMPAT:       %[[VALUE:[0-9]+]] = load i8, ptr %v.addr, align 1
// CHECK-MSCOMPAT:       store atomic volatile i8 %[[VALUE]], ptr %[[DEST]] release, align 1
// CHECK-MSCOMPAT:       ret void
// CHECK-LINUX: error: call to undeclared function '__stlr8'

void test__stlr16(unsigned __int16 volatile *p, unsigned __int16 v)
{
  __stlr16 (p, v);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}void @test__stlr16(ptr{{.*}}%p, i16{{.*}}%v){{.*}}{
// CHECK-MSCOMPAT:       %[[DEST:[0-9]+]] = load ptr, ptr %p.addr, align 8
// CHECK-MSCOMPAT:       %[[VALUE:[0-9]+]] = load i16, ptr %v.addr, align 2
// CHECK-MSCOMPAT:       store atomic volatile i16 %[[VALUE]], ptr %[[DEST]] release, align 2
// CHECK-MSCOMPAT:       ret void
// CHECK-LINUX: error: call to undeclared function '__stlr16'

void test__stlr32(unsigned __int32 volatile *p, unsigned __int32 v)
{
  __stlr32(p, v);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}void @test__stlr32(ptr{{.*}}%p, i32{{.*}}%v){{.*}}{
// CHECK-MSCOMPAT:       %[[DEST:[0-9]+]] = load ptr, ptr %p.addr, align 8
// CHECK-MSCOMPAT:       %[[VALUE:[0-9]+]] = load i32, ptr %v.addr, align 4
// CHECK-MSCOMPAT:       store atomic volatile i32 %[[VALUE]], ptr %[[DEST]] release, align 4
// CHECK-MSCOMPAT:       ret void
// CHECK-LINUX: error: call to undeclared function '__stlr32'

void test__stlr64(unsigned __int64 volatile *p, unsigned __int64 v)
{
  __stlr64(p, v);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}void @test__stlr64(ptr{{.*}}%p, i64{{.*}}%v){{.*}}{
// CHECK-MSCOMPAT:       %[[DEST:[0-9]+]] = load ptr, ptr %p.addr, align 8
// CHECK-MSCOMPAT:       %[[VALUE:[0-9]+]] = load i64, ptr %v.addr, align 8
// CHECK-MSCOMPAT:       store atomic volatile i64 %[[VALUE]], ptr %[[DEST]] release, align 8
// CHECK-MSCOMPAT:       ret void
// CHECK-LINUX: error: call to undeclared function '__stlr64'

unsigned char test__cas8(unsigned char volatile* t, unsigned char c, unsigned char v)
{
  return __cas8 (t, c, v);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i8 @test__cas8(ptr{{.*}}%t, i8{{.*}}%c, i8{{.*}}%v){{.*}}{
// CHECK-MSCOMPAT:       %[[TMPT:[0-9]+]] = load ptr, ptr %t.addr, align 8
// CHECK-MSCOMPAT:       %[[TMPC:[0-9]+]] = load i8, ptr %c.addr, align 1
// CHECK-MSCOMPAT:       %[[TMPV:[0-9]+]] = load i8, ptr %v.addr, align 1
// CHECK-MSCOMPAT:       %[[ZEXTC:[0-9]+]] = zext i8 %[[TMPC]] to i32
// CHECK-MSCOMPAT:       %[[ZEXTV:[0-9]+]]  = zext i8 %[[TMPV]] to i32
// CHECK-MSCOMPAT:       %[[RET:[0-9]+]] = call i32 @llvm.aarch64.cas8(ptr %[[TMPT]], i32 %[[ZEXTC]], i32 %[[ZEXTV]])
// CHECK-MSCOMPAT:       %[[RETT:[0-9]+]]  = trunc i32 %[[RET]] to i8
// CHECK-MSCOMPAT:       ret i8 %[[RETT]]
// CHECK-LINUX: error: call to undeclared function '__cas8'

unsigned short test__cas16(unsigned short volatile* t, unsigned short c, unsigned short v)
{
  return __cas16 (t, c, v);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i16 @test__cas16(ptr{{.*}}%t, i16{{.*}}%c, i16{{.*}}%v){{.*}}{
// CHECK-MSCOMPAT:       %[[TMPT:[0-9]+]] = load ptr, ptr %t.addr, align 8
// CHECK-MSCOMPAT:       %[[TMPC:[0-9]+]] = load i16, ptr %c.addr, align 2
// CHECK-MSCOMPAT:       %[[TMPV:[0-9]+]] = load i16, ptr %v.addr, align 2
// CHECK-MSCOMPAT:       %[[ZEXTC:[0-9]+]] = zext i16 %[[TMPC]] to i32
// CHECK-MSCOMPAT:       %[[ZEXTV:[0-9]+]]  = zext i16 %[[TMPV]] to i32 
// CHECK-MSCOMPAT:       %[[RET:[0-9]+]] = call i32 @llvm.aarch64.cas16(ptr %[[TMPT]], i32 %[[ZEXTC]], i32 %[[ZEXTV]])
// CHECK-MSCOMPAT:       %[[RETT:[0-9]+]]  = trunc i32 %[[RET]] to i16
// CHECK-MSCOMPAT:       ret i16 %[[RETT]]
// CHECK-LINUX: error: call to undeclared function '__cas16'

unsigned int test__cas32(unsigned int volatile* t, unsigned int c, unsigned int v)
{
  return __cas32 (t, c, v);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i32 @test__cas32(ptr{{.*}}%t, i32{{.*}}%c, i32{{.*}}%v){{.*}}{
// CHECK-MSCOMPAT:       %[[TMPT:[0-9]+]] = load ptr, ptr %t.addr, align 8
// CHECK-MSCOMPAT:       %[[TMPC:[0-9]+]] = load i32, ptr %c.addr, align 4
// CHECK-MSCOMPAT:       %[[TMPV:[0-9]+]] = load i32, ptr %v.addr, align 4
// CHECK-MSCOMPAT:       %[[RET:[0-9]+]] = call i32 @llvm.aarch64.cas32(ptr %[[TMPT]], i32 %[[TMPC]], i32 %[[TMPV]])
// CHECK-MSCOMPAT:       ret i32 %[[RET]]
// CHECK-LINUX: error: call to undeclared function '__cas32'

unsigned long long int test__cas64(unsigned long long int volatile* t,
                                   unsigned long long int c,
                                   unsigned long long int v)
{
  return __cas64 (t, c, v);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i64 @test__cas64(ptr{{.*}}%t, i64{{.*}}%c, i64{{.*}}%v){{.*}}{
// CHECK-MSCOMPAT:       %[[TMPT:[0-9]+]] = load ptr, ptr %t.addr, align 8
// CHECK-MSCOMPAT:       %[[TMPC:[0-9]+]] = load i64, ptr %c.addr, align 8
// CHECK-MSCOMPAT:       %[[TMPV:[0-9]+]] = load i64, ptr %v.addr, align 8
// CHECK-MSCOMPAT:       %[[RET:[0-9]+]] = call i64 @llvm.aarch64.cas64(ptr %[[TMPT]], i64 %[[TMPC]], i64 %[[TMPV]])
// CHECK-MSCOMPAT:       ret i64 %[[RET]]
// CHECK-LINUX: error: call to undeclared function '__cas64'

unsigned char test__casa8(unsigned char volatile* t, unsigned char c, unsigned char v)
{
  return __casa8 (t, c, v);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i8 @test__casa8(ptr{{.*}}%t, i8{{.*}}%c, i8{{.*}}%v){{.*}}{
// CHECK-MSCOMPAT:       %[[TMPT:[0-9]+]] = load ptr, ptr %t.addr, align 8
// CHECK-MSCOMPAT:       %[[TMPC:[0-9]+]] = load i8, ptr %c.addr, align 1
// CHECK-MSCOMPAT:       %[[TMPV:[0-9]+]] = load i8, ptr %v.addr, align 1
// CHECK-MSCOMPAT:       %[[ZEXTC:[0-9]+]] = zext i8 %[[TMPC]] to i32
// CHECK-MSCOMPAT:       %[[ZEXTV:[0-9]+]]  = zext i8 %[[TMPV]] to i32
// CHECK-MSCOMPAT:       %[[RET:[0-9]+]] = call i32 @llvm.aarch64.casa8(ptr %[[TMPT]], i32 %[[ZEXTC]], i32 %[[ZEXTV]])
// CHECK-MSCOMPAT:       %[[RETT:[0-9]+]]  = trunc i32 %[[RET]] to i8
// CHECK-MSCOMPAT:       ret i8 %[[RETT]]
// CHECK-LINUX: error: call to undeclared function '__casa8'

unsigned short test__casa16(unsigned short volatile* t, unsigned short c, unsigned short v)
{
  return __casa16 (t, c, v);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i16 @test__casa16(ptr{{.*}}%t, i16{{.*}}%c, i16{{.*}}%v){{.*}}{
// CHECK-MSCOMPAT:       %[[TMPT:[0-9]+]] = load ptr, ptr %t.addr, align 8
// CHECK-MSCOMPAT:       %[[TMPC:[0-9]+]] = load i16, ptr %c.addr, align 2
// CHECK-MSCOMPAT:       %[[TMPV:[0-9]+]] = load i16, ptr %v.addr, align 2
// CHECK-MSCOMPAT:       %[[ZEXTC:[0-9]+]] = zext i16 %[[TMPC]] to i32
// CHECK-MSCOMPAT:       %[[ZEXTV:[0-9]+]]  = zext i16 %[[TMPV]] to i32
// CHECK-MSCOMPAT:       %[[RET:[0-9]+]] = call i32 @llvm.aarch64.casa16(ptr %[[TMPT]], i32 %[[ZEXTC]], i32 %[[ZEXTV]])
// CHECK-MSCOMPAT:       %[[RETT:[0-9]+]]  = trunc i32 %[[RET]] to i16
// CHECK-MSCOMPAT:       ret i16 %[[RETT]]
// CHECK-LINUX: error: call to undeclared function '__casa16'

unsigned int test__casa32(unsigned int volatile* t, unsigned int c, unsigned int v)
{
  return __casa32 (t, c, v);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i32 @test__casa32(ptr{{.*}}%t, i32{{.*}}%c, i32{{.*}}%v){{.*}}{
// CHECK-MSCOMPAT:       %[[TMPT:[0-9]+]] = load ptr, ptr %t.addr, align 8
// CHECK-MSCOMPAT:       %[[TMPC:[0-9]+]] = load i32, ptr %c.addr, align 4
// CHECK-MSCOMPAT:       %[[TMPV:[0-9]+]] = load i32, ptr %v.addr, align 4
// CHECK-MSCOMPAT:       %[[RET:[0-9]+]] = call i32 @llvm.aarch64.casa32(ptr %[[TMPT]], i32 %[[TMPC]], i32 %[[TMPV]])
// CHECK-MSCOMPAT:       ret i32 %[[RET]]
// CHECK-LINUX: error: call to undeclared function '__casa32'

unsigned long long int test__casa64(unsigned long long int volatile* t,
                                    unsigned long long int c,
                                    unsigned long long int v)
{
  return __casa64 (t, c, v);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i64 @test__casa64(ptr{{.*}}%t, i64{{.*}}%c, i64{{.*}}%v){{.*}}{
// CHECK-MSCOMPAT:       %[[TMPT:[0-9]+]] = load ptr, ptr %t.addr, align 8
// CHECK-MSCOMPAT:       %[[TMPC:[0-9]+]] = load i64, ptr %c.addr, align 8
// CHECK-MSCOMPAT:       %[[TMPV:[0-9]+]] = load i64, ptr %v.addr, align 8
// CHECK-MSCOMPAT:       %[[RET:[0-9]+]] = call i64 @llvm.aarch64.casa64(ptr %[[TMPT]], i64 %[[TMPC]], i64 %[[TMPV]])
// CHECK-MSCOMPAT:       ret i64 %[[RET]]
// CHECK-LINUX: error: call to undeclared function '__casa64'

unsigned char test__casl8(unsigned char volatile* t, unsigned char c, unsigned char v)
{
  return __casl8 (t, c, v);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i8 @test__casl8(ptr{{.*}}%t, i8{{.*}}%c, i8{{.*}}%v){{.*}}{
// CHECK-MSCOMPAT:       %[[TMPT:[0-9]+]] = load ptr, ptr %t.addr, align 8
// CHECK-MSCOMPAT:       %[[TMPC:[0-9]+]] = load i8, ptr %c.addr, align 1
// CHECK-MSCOMPAT:       %[[TMPV:[0-9]+]] = load i8, ptr %v.addr, align 1
// CHECK-MSCOMPAT:       %[[ZEXTC:[0-9]+]] = zext i8 %[[TMPC]] to i32
// CHECK-MSCOMPAT:       %[[ZEXTV:[0-9]+]]  = zext i8 %[[TMPV]] to i32
// CHECK-MSCOMPAT:       %[[RET:[0-9]+]] = call i32 @llvm.aarch64.casl8(ptr %[[TMPT]], i32 %[[ZEXTC]], i32 %[[ZEXTV]])
// CHECK-MSCOMPAT:       %[[RETT:[0-9]+]]  = trunc i32 %[[RET]] to i8
// CHECK-MSCOMPAT:       ret i8 %[[RETT]]
// CHECK-LINUX: error: call to undeclared function '__casl8'

unsigned short test__casl16(unsigned short volatile* t, unsigned short c, unsigned short v)
{
  return __casl16 (t, c, v);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i16 @test__casl16(ptr{{.*}}%t, i16{{.*}}%c, i16{{.*}}%v){{.*}}{
// CHECK-MSCOMPAT:       %[[TMPT:[0-9]+]] = load ptr, ptr %t.addr, align 8
// CHECK-MSCOMPAT:       %[[TMPC:[0-9]+]] = load i16, ptr %c.addr, align 2
// CHECK-MSCOMPAT:       %[[TMPV:[0-9]+]] = load i16, ptr %v.addr, align 2
// CHECK-MSCOMPAT:       %[[ZEXTC:[0-9]+]] = zext i16 %[[TMPC]] to i32
// CHECK-MSCOMPAT:       %[[ZEXTV:[0-9]+]]  = zext i16 %[[TMPV]] to i32
// CHECK-MSCOMPAT:       %[[RET:[0-9]+]] = call i32 @llvm.aarch64.casl16(ptr %[[TMPT]], i32 %[[ZEXTC]], i32 %[[ZEXTV]])
// CHECK-MSCOMPAT:       %[[RETT:[0-9]+]]  = trunc i32 %[[RET]] to i16
// CHECK-MSCOMPAT:       ret i16 %[[RETT]]
// CHECK-LINUX: error: call to undeclared function '__casl16'

unsigned int test__casl32(unsigned int volatile* t, unsigned int c, unsigned int v)
{
  return __casl32 (t, c, v);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i32 @test__casl32(ptr{{.*}}%t, i32{{.*}}%c, i32{{.*}}%v){{.*}}{
// CHECK-MSCOMPAT:       %[[TMPT:[0-9]+]] = load ptr, ptr %t.addr, align 8
// CHECK-MSCOMPAT:       %[[TMPC:[0-9]+]] = load i32, ptr %c.addr, align 4
// CHECK-MSCOMPAT:       %[[TMPV:[0-9]+]] = load i32, ptr %v.addr, align 4
// CHECK-MSCOMPAT:       %[[RET:[0-9]+]] = call i32 @llvm.aarch64.casl32(ptr %[[TMPT]], i32 %[[TMPC]], i32 %[[TMPV]])
// CHECK-MSCOMPAT:       ret i32 %[[RET]]
// CHECK-LINUX: error: call to undeclared function '__casl32'

unsigned long long int test__casl64(unsigned long long int volatile* t,
                                    unsigned long long int c,
                                    unsigned long long int v)
{
  return __casl64 (t, c, v);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i64 @test__casl64(ptr{{.*}}%t, i64{{.*}}%c, i64{{.*}}%v){{.*}}{
// CHECK-MSCOMPAT:       %[[TMPT:[0-9]+]] = load ptr, ptr %t.addr, align 8
// CHECK-MSCOMPAT:       %[[TMPC:[0-9]+]] = load i64, ptr %c.addr, align 8
// CHECK-MSCOMPAT:       %[[TMPV:[0-9]+]] = load i64, ptr %v.addr, align 8
// CHECK-MSCOMPAT:       %[[RET:[0-9]+]] = call i64 @llvm.aarch64.casl64(ptr %[[TMPT]], i64 %[[TMPC]], i64 %[[TMPV]])
// CHECK-MSCOMPAT:       ret i64 %[[RET]]
// CHECK-LINUX: error: call to undeclared function '__casl64'

unsigned char test__casal8(unsigned char volatile* t, unsigned char c, unsigned char v)
{
  return __casal8 (t, c, v);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i8 @test__casal8(ptr{{.*}}%t, i8{{.*}}%c, i8{{.*}}%v){{.*}}{
// CHECK-MSCOMPAT:       %[[TMPT:[0-9]+]] = load ptr, ptr %t.addr, align 8
// CHECK-MSCOMPAT:       %[[TMPC:[0-9]+]] = load i8, ptr %c.addr, align 1
// CHECK-MSCOMPAT:       %[[TMPV:[0-9]+]] = load i8, ptr %v.addr, align 1
// CHECK-MSCOMPAT:       %[[ZEXTC:[0-9]+]] = zext i8 %[[TMPC]] to i32
// CHECK-MSCOMPAT:       %[[ZEXTV:[0-9]+]]  = zext i8 %[[TMPV]] to i32
// CHECK-MSCOMPAT:       %[[RET:[0-9]+]] = call i32 @llvm.aarch64.casal8(ptr %[[TMPT]], i32 %[[ZEXTC]], i32 %[[ZEXTV]])
// CHECK-MSCOMPAT:       %[[RETT:[0-9]+]]  = trunc i32 %[[RET]] to i8
// CHECK-MSCOMPAT:       ret i8 %[[RETT]]
// CHECK-LINUX: error: call to undeclared function '__casal8'

unsigned short test__casal16(unsigned short volatile* t, unsigned short c, unsigned short v)
{
  return __casal16 (t, c, v);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i16 @test__casal16(ptr{{.*}}%t, i16{{.*}}%c, i16{{.*}}%v){{.*}}{
// CHECK-MSCOMPAT:       %[[TMPT:[0-9]+]] = load ptr, ptr %t.addr, align 8
// CHECK-MSCOMPAT:       %[[TMPC:[0-9]+]] = load i16, ptr %c.addr, align 2
// CHECK-MSCOMPAT:       %[[TMPV:[0-9]+]] = load i16, ptr %v.addr, align 2
// CHECK-MSCOMPAT:       %[[ZEXTC:[0-9]+]] = zext i16 %[[TMPC]] to i32
// CHECK-MSCOMPAT:       %[[ZEXTV:[0-9]+]]  = zext i16 %[[TMPV]] to i32
// CHECK-MSCOMPAT:       %[[RET:[0-9]+]] = call i32 @llvm.aarch64.casal16(ptr %[[TMPT]], i32 %[[ZEXTC]], i32 %[[ZEXTV]])
// CHECK-MSCOMPAT:       %[[RETT:[0-9]+]]  = trunc i32 %[[RET]] to i16
// CHECK-MSCOMPAT:       ret i16 %[[RETT]]
// CHECK-LINUX: error: call to undeclared function '__casal16'

unsigned int test__casal32(unsigned int volatile* t, unsigned int c, unsigned int v)
{
  return __casal32 (t, c, v);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i32 @test__casal32(ptr{{.*}}%t, i32{{.*}}%c, i32{{.*}}%v){{.*}}{
// CHECK-MSCOMPAT:       %[[TMPT:[0-9]+]] = load ptr, ptr %t.addr, align 8
// CHECK-MSCOMPAT:       %[[TMPC:[0-9]+]] = load i32, ptr %c.addr, align 4
// CHECK-MSCOMPAT:       %[[TMPV:[0-9]+]] = load i32, ptr %v.addr, align 4
// CHECK-MSCOMPAT:       %[[RET:[0-9]+]] = call i32 @llvm.aarch64.casal32(ptr %[[TMPT]], i32 %[[TMPC]], i32 %[[TMPV]])
// CHECK-MSCOMPAT:       ret i32 %[[RET]]
// CHECK-LINUX: error: call to undeclared function '__casal32'

unsigned long long int test__casal64(unsigned long long int volatile* t,
                                     unsigned long long int c,
                                     unsigned long long int v)
{
  return __casal64 (t, c, v);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i64 @test__casal64(ptr{{.*}}%t, i64{{.*}}%c, i64{{.*}}%v){{.*}}{
// CHECK-MSCOMPAT:       %[[TMPT:[0-9]+]] = load ptr, ptr %t.addr, align 8
// CHECK-MSCOMPAT:       %[[TMPC:[0-9]+]] = load i64, ptr %c.addr, align 8
// CHECK-MSCOMPAT:       %[[TMPV:[0-9]+]] = load i64, ptr %v.addr, align 8
// CHECK-MSCOMPAT:       %[[RET:[0-9]+]] = call i64 @llvm.aarch64.casal64(ptr %[[TMPT]], i64 %[[TMPC]], i64 %[[TMPV]])
// CHECK-MSCOMPAT:       ret i64 %[[RET]]
// CHECK-LINUX: error: call to undeclared function '__casal64'

unsigned char test__swp8(unsigned char volatile* t, unsigned char v)
{
  return __swp8(t, v);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i8 @test__swp8(ptr{{.*}}%t, i8{{.*}}%v){{.*}}{
// CHECK-MSCOMPAT:       %[[TMPT:[0-9]+]] = load ptr, ptr %t.addr, align 8
// CHECK-MSCOMPAT:       %[[TMPV:[0-9]+]] = load i8, ptr %v.addr, align 1
// CHECK-MSCOMPAT:       %[[ZEXTV:[0-9]+]] = zext i8 %[[TMPV]] to i32
// CHECK-MSCOMPAT:       %[[RET:[0-9]+]] = call i32 @llvm.aarch64.swp8(ptr %[[TMPT]], i32 %[[ZEXTV]])
// CHECK-MSCOMPAT:       %[[TRUNC:[0-9]+]] = trunc i32 %[[RET]] to i8
// CHECK-MSCOMPAT:       ret i8 %[[TRUNC]]
// CHECK-LINUX: error: call to undeclared function '__swp8'

unsigned short test__swp16(unsigned short volatile* t, unsigned short v)
{
  return __swp16(t, v);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i16 @test__swp16(ptr{{.*}}%t, i16{{.*}}%v){{.*}}{
// CHECK-MSCOMPAT:       %[[TMPT:[0-9]+]] = load ptr, ptr %t.addr, align 8
// CHECK-MSCOMPAT:       %[[TMPV:[0-9]+]] = load i16, ptr %v.addr, align 2
// CHECK-MSCOMPAT:       %[[ZEXTV:[0-9]+]] = zext i16 %[[TMPV]] to i32
// CHECK-MSCOMPAT:       %[[RET:[0-9]+]] = call i32 @llvm.aarch64.swp16(ptr %[[TMPT]], i32 %[[ZEXTV]])
// CHECK-MSCOMPAT:       %[[TRUNC:[0-9]+]] = trunc i32 %[[RET]] to i16
// CHECK-MSCOMPAT:       ret i16 %[[TRUNC]]
// CHECK-LINUX: error: call to undeclared function '__swp16'

unsigned int test__swp32(unsigned int volatile* t, unsigned int v)
{
  return __swp32(t, v);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i32 @test__swp32(ptr{{.*}}%t, i32{{.*}}%v){{.*}}{
// CHECK-MSCOMPAT:       %[[TMPT:[0-9]+]] = load ptr, ptr %t.addr, align 8
// CHECK-MSCOMPAT:       %[[TMPV:[0-9]+]] = load i32, ptr %v.addr, align 4
// CHECK-MSCOMPAT:       %[[RET:[0-9]+]] = call i32 @llvm.aarch64.swp32(ptr %[[TMPT]], i32 %[[TMPV]])
// CHECK-MSCOMPAT:       ret i32 %[[RET]]
// CHECK-LINUX: error: call to undeclared function '__swp32'

unsigned long long int test__swp64(unsigned long long int volatile* t,
                                   unsigned long long int v)
{
  return __swp64(t, v);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i64 @test__swp64(ptr{{.*}}%t, i64{{.*}}%v){{.*}}{
// CHECK-MSCOMPAT:       %[[TMPT:[0-9]+]] = load ptr, ptr %t.addr, align 8
// CHECK-MSCOMPAT:       %[[TMPV:[0-9]+]] = load i64, ptr %v.addr, align 8
// CHECK-MSCOMPAT:       %[[RET:[0-9]+]] = call i64 @llvm.aarch64.swp64(ptr %[[TMPT]], i64 %[[TMPV]])
// CHECK-MSCOMPAT:       ret i64 %[[RET]]
// CHECK-LINUX: error: call to undeclared function '__swp64'

unsigned char test__swpa8(unsigned char volatile* t, unsigned char v)
{
  return __swpa8(t, v);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i8 @test__swpa8(ptr{{.*}}%t, i8{{.*}}%v){{.*}}{
// CHECK-MSCOMPAT:       %[[TMPT:[0-9]+]] = load ptr, ptr %t.addr, align 8
// CHECK-MSCOMPAT:       %[[TMPV:[0-9]+]] = load i8, ptr %v.addr, align 1
// CHECK-MSCOMPAT:       %[[ZEXTV:[0-9]+]] = zext i8 %[[TMPV]] to i32
// CHECK-MSCOMPAT:       %[[RET:[0-9]+]] = call i32 @llvm.aarch64.swpa8(ptr %[[TMPT]], i32 %[[ZEXTV]])
// CHECK-MSCOMPAT:       %[[TRUNC:[0-9]+]] = trunc i32 %[[RET]] to i8
// CHECK-MSCOMPAT:       ret i8 %[[TRUNC]]
// CHECK-LINUX: error: call to undeclared function '__swpa8'

unsigned short test__swpa16(unsigned short volatile* t, unsigned short v)
{
  return __swpa16(t, v);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i16 @test__swpa16(ptr{{.*}}%t, i16{{.*}}%v){{.*}}{
// CHECK-MSCOMPAT:       %[[TMPT:[0-9]+]] = load ptr, ptr %t.addr, align 8
// CHECK-MSCOMPAT:       %[[TMPV:[0-9]+]] = load i16, ptr %v.addr, align 2
// CHECK-MSCOMPAT:       %[[ZEXTV:[0-9]+]] = zext i16 %[[TMPV]] to i32
// CHECK-MSCOMPAT:       %[[RET:[0-9]+]] = call i32 @llvm.aarch64.swpa16(ptr %[[TMPT]], i32 %[[ZEXTV]])
// CHECK-MSCOMPAT:       %[[TRUNC:[0-9]+]] = trunc i32 %[[RET]] to i16
// CHECK-MSCOMPAT:       ret i16 %[[TRUNC]]
// CHECK-LINUX: error: call to undeclared function '__swpa16'

unsigned int test__swpa32(unsigned int volatile* t, unsigned int v)
{
  return __swpa32(t, v);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i32 @test__swpa32(ptr{{.*}}%t, i32{{.*}}%v){{.*}}{
// CHECK-MSCOMPAT:       %[[TMPT:[0-9]+]] = load ptr, ptr %t.addr, align 8
// CHECK-MSCOMPAT:       %[[TMPV:[0-9]+]] = load i32, ptr %v.addr, align 4
// CHECK-MSCOMPAT:       %[[RET:[0-9]+]] = call i32 @llvm.aarch64.swpa32(ptr %[[TMPT]], i32 %[[TMPV]])
// CHECK-MSCOMPAT:       ret i32 %[[RET]]
// CHECK-LINUX: error: call to undeclared function '__swpa32'

unsigned long long int test__swpa64(unsigned long long int volatile* t,
                                    unsigned long long int v)
{
  return __swpa64(t, v);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i64 @test__swpa64(ptr{{.*}}%t, i64{{.*}}%v){{.*}}{
// CHECK-MSCOMPAT:       %[[TMPT:[0-9]+]] = load ptr, ptr %t.addr, align 8
// CHECK-MSCOMPAT:       %[[TMPV:[0-9]+]] = load i64, ptr %v.addr, align 8
// CHECK-MSCOMPAT:       %[[RET:[0-9]+]] = call i64 @llvm.aarch64.swpa64(ptr %[[TMPT]], i64 %[[TMPV]])
// CHECK-MSCOMPAT:       ret i64 %[[RET]]
// CHECK-LINUX: error: call to undeclared function '__swpa64'

unsigned char test__swpl8(unsigned char volatile* t, unsigned char v)
{
  return __swpl8(t, v);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i8 @test__swpl8(ptr{{.*}}%t, i8{{.*}}%v){{.*}}{
// CHECK-MSCOMPAT:       %[[TMPT:[0-9]+]] = load ptr, ptr %t.addr, align 8
// CHECK-MSCOMPAT:       %[[TMPV:[0-9]+]] = load i8, ptr %v.addr, align 1
// CHECK-MSCOMPAT:       %[[ZEXTV:[0-9]+]] = zext i8 %[[TMPV]] to i32
// CHECK-MSCOMPAT:       %[[RET:[0-9]+]] = call i32 @llvm.aarch64.swpl8(ptr %[[TMPT]], i32 %[[ZEXTV]])
// CHECK-MSCOMPAT:       %[[TRUNC:[0-9]+]] = trunc i32 %[[RET]] to i8
// CHECK-MSCOMPAT:       ret i8 %[[TRUNC]]
// CHECK-LINUX: error: call to undeclared function '__swpl8'

unsigned short test__swpl16(unsigned short volatile* t, unsigned short v)
{
  return __swpl16(t, v);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i16 @test__swpl16(ptr{{.*}}%t, i16{{.*}}%v){{.*}}{
// CHECK-MSCOMPAT:       %[[TMPT:[0-9]+]] = load ptr, ptr %t.addr, align 8
// CHECK-MSCOMPAT:       %[[TMPV:[0-9]+]] = load i16, ptr %v.addr, align 2
// CHECK-MSCOMPAT:       %[[ZEXTV:[0-9]+]] = zext i16 %[[TMPV]] to i32
// CHECK-MSCOMPAT:       %[[RET:[0-9]+]] = call i32 @llvm.aarch64.swpl16(ptr %[[TMPT]], i32 %[[ZEXTV]])
// CHECK-MSCOMPAT:       %[[TRUNC:[0-9]+]] = trunc i32 %[[RET]] to i16
// CHECK-MSCOMPAT:       ret i16 %[[TRUNC]]
// CHECK-LINUX: error: call to undeclared function '__swpl16'

unsigned int test__swpl32(unsigned int volatile* t, unsigned int v)
{
  return __swpl32(t, v);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i32 @test__swpl32(ptr{{.*}}%t, i32{{.*}}%v){{.*}}{
// CHECK-MSCOMPAT:       %[[TMPT:[0-9]+]] = load ptr, ptr %t.addr, align 8
// CHECK-MSCOMPAT:       %[[TMPV:[0-9]+]] = load i32, ptr %v.addr, align 4
// CHECK-MSCOMPAT:       %[[RET:[0-9]+]] = call i32 @llvm.aarch64.swpl32(ptr %[[TMPT]], i32 %[[TMPV]])
// CHECK-MSCOMPAT:       ret i32 %[[RET]]
// CHECK-LINUX: error: call to undeclared function '__swpl32'

unsigned long long int test__swpl64(unsigned long long int volatile* t,
                                    unsigned long long int v)
{
  return __swpl64(t, v);
}
// CHECK-MSCOMPAT-LABEL: define{{.*}}i64 @test__swpl64(ptr{{.*}}%t, i64{{.*}}%v){{.*}}{
// CHECK-MSCOMPAT:       %[[TMPT:[0-9]+]] = load ptr, ptr %t.addr, align 8
// CHECK-MSCOMPAT:       %[[TMPV:[0-9]+]] = load i64, ptr %v.addr, align 8
// CHECK-MSCOMPAT:       %[[RET:[0-9]+]] = call i64 @llvm.aarch64.swpl64(ptr %[[TMPT]], i64 %[[TMPV]])
// CHECK-MSCOMPAT:       ret i64 %[[RET]]
// CHECK-LINUX: error: call to undeclared function '__swpl64'

// CHECK-MSCOMPAT: ![[MD2]] = !{!"x18"}
// CHECK-MSCOMPAT: ![[MD3]] = !{!"sp"}
// CHECK-MSCOMPAT: ![[MD4]] = !{!"d5"}
// CHECK-MSCOMPAT: ![[MD5]] = !{!"d31"}
