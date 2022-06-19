// RUN: %clang_cc1 -no-opaque-pointers -triple arm64-windows -Wno-implicit-function-declaration -fms-compatibility -emit-llvm -o - %s \
// RUN:    | FileCheck %s -check-prefix CHECK-MSVC

// RUN: not %clang_cc1 -no-opaque-pointers -triple arm64-linux -Werror -S -o /dev/null %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-LINUX

long test_InterlockedAdd(long volatile *Addend, long Value) {
  return _InterlockedAdd(Addend, Value);
}

long test_InterlockedAdd_constant(long volatile *Addend) {
  return _InterlockedAdd(Addend, -1);
}

// CHECK-LABEL: define {{.*}} i32 @test_InterlockedAdd(i32* %Addend, i32 %Value) {{.*}} {
// CHECK-MSVC: %[[OLDVAL:[0-9]+]] = atomicrmw add i32* %1, i32 %2 seq_cst, align 4
// CHECK-MSVC: %[[NEWVAL:[0-9]+]] = add i32 %[[OLDVAL:[0-9]+]], %2
// CHECK-MSVC: ret i32 %[[NEWVAL:[0-9]+]]
// CHECK-LINUX: error: call to undeclared function '_InterlockedAdd'

void check__dmb(void) {
  __dmb(0);
}

// CHECK-MSVC: @llvm.aarch64.dmb(i32 0)
// CHECK-LINUX: error: call to undeclared function '__dmb'

void check__dsb(void) {
  __dsb(0);
}

// CHECK-MSVC: @llvm.aarch64.dsb(i32 0)
// CHECK-LINUX: error: call to undeclared function '__dsb'

void check__isb(void) {
  __isb(0);
}

// CHECK-MSVC: @llvm.aarch64.isb(i32 0)
// CHECK-LINUX: error: call to undeclared function '__isb'

void check__yield(void) {
  __yield();
}

// CHECK-MSVC: @llvm.aarch64.hint(i32 1)
// CHECK-LINUX: error: call to undeclared function '__yield'

void check__wfe(void) {
  __wfe();
}

// CHECK-MSVC: @llvm.aarch64.hint(i32 2)
// CHECK-LINUX: error: call to undeclared function '__wfe'

void check__wfi(void) {
  __wfi();
}

// CHECK-MSVC: @llvm.aarch64.hint(i32 3)
// CHECK-LINUX: error: call to undeclared function '__wfi'

void check__sev(void) {
  __sev();
}

// CHECK-MSVC: @llvm.aarch64.hint(i32 4)
// CHECK-LINUX: error: call to undeclared function '__sev'

void check__sevl(void) {
  __sevl();
}

// CHECK-MSVC: @llvm.aarch64.hint(i32 5)
// CHECK-LINUX: error: call to undeclared function '__sevl'

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

unsigned __int64 check__getReg(void) {
  unsigned volatile __int64 reg;
  reg = __getReg(18);
  reg = __getReg(31);
  return reg;
}

// CHECK-MSVC: call i64 @llvm.read_register.i64(metadata ![[MD2:.*]])
// CHECK-MSVC: call i64 @llvm.read_register.i64(metadata ![[MD3:.*]])

void check__writex18byte(unsigned long offset, unsigned char data) {
  __writex18byte(offset, data);
}

// CHECK-MSVC: %[[DATA_ADDR:.*]] = alloca i8, align 1
// CHECK-MSVC: %[[OFFSET_ADDR:.*]] = alloca i32, align 4
// CHECK-MSVC: store i8 %data, i8* %[[DATA_ADDR]], align 1
// CHECK-MSVC: store i32 %offset, i32* %[[OFFSET_ADDR]], align 4
// CHECK-MSVC: %[[X18:.*]] = call i64 @llvm.read_register.i64(metadata ![[MD2]])
// CHECK-MSVC: %[[X18_AS_PTR:.*]] = inttoptr i64 %[[X18]] to i8*
// CHECK-MSVC: %[[OFFSET:.*]] = load i32, i32* %[[OFFSET_ADDR]], align 4
// CHECK-MSVC: %[[ZEXT_OFFSET:.*]] = zext i32 %[[OFFSET]] to i64
// CHECK-MSVC: %[[PTR:.*]] = getelementptr i8, i8* %[[X18_AS_PTR]], i64 %[[ZEXT_OFFSET]]
// CHECK-MSVC: %[[DATA:.*]] = load i8, i8* %[[DATA_ADDR]], align 1
// CHECK-MSVC: store i8 %[[DATA]], i8* %[[PTR]], align 1

void check__writex18word(unsigned long offset, unsigned short data) {
  __writex18word(offset, data);
}

// CHECK-MSVC: %[[DATA_ADDR:.*]] = alloca i16, align 2
// CHECK-MSVC: %[[OFFSET_ADDR:.*]] = alloca i32, align 4
// CHECK-MSVC: store i16 %data, i16* %[[DATA_ADDR]], align 2
// CHECK-MSVC: store i32 %offset, i32* %[[OFFSET_ADDR]], align 4
// CHECK-MSVC: %[[X18:.*]] = call i64 @llvm.read_register.i64(metadata ![[MD2]])
// CHECK-MSVC: %[[X18_AS_PTR:.*]] = inttoptr i64 %[[X18]] to i8*
// CHECK-MSVC: %[[OFFSET:.*]] = load i32, i32* %[[OFFSET_ADDR]], align 4
// CHECK-MSVC: %[[ZEXT_OFFSET:.*]] = zext i32 %[[OFFSET]] to i64
// CHECK-MSVC: %[[PTR:.*]] = getelementptr i8, i8* %[[X18_AS_PTR]], i64 %[[ZEXT_OFFSET]]
// CHECK-MSVC: %[[BITCAST_PTR:.*]] = bitcast i8* %[[PTR]] to i16*
// CHECK-MSVC: %[[DATA:.*]] = load i16, i16* %[[DATA_ADDR]], align 2
// CHECK-MSVC: store i16 %[[DATA]], i16* %[[BITCAST_PTR]], align 1

void check__writex18dword(unsigned long offset, unsigned long data) {
  __writex18dword(offset, data);
}

// CHECK-MSVC: %[[DATA_ADDR:.*]] = alloca i32, align 4
// CHECK-MSVC: %[[OFFSET_ADDR:.*]] = alloca i32, align 4
// CHECK-MSVC: store i32 %data, i32* %[[DATA_ADDR]], align 4
// CHECK-MSVC: store i32 %offset, i32* %[[OFFSET_ADDR]], align 4
// CHECK-MSVC: %[[X18:.*]] = call i64 @llvm.read_register.i64(metadata ![[MD2]])
// CHECK-MSVC: %[[X18_AS_PTR:.*]] = inttoptr i64 %[[X18]] to i8*
// CHECK-MSVC: %[[OFFSET:.*]] = load i32, i32* %[[OFFSET_ADDR]], align 4
// CHECK-MSVC: %[[ZEXT_OFFSET:.*]] = zext i32 %[[OFFSET]] to i64
// CHECK-MSVC: %[[PTR:.*]] = getelementptr i8, i8* %[[X18_AS_PTR]], i64 %[[ZEXT_OFFSET]]
// CHECK-MSVC: %[[BITCAST_PTR:.*]] = bitcast i8* %[[PTR]] to i32*
// CHECK-MSVC: %[[DATA:.*]] = load i32, i32* %[[DATA_ADDR]], align 4
// CHECK-MSVC: store i32 %[[DATA]], i32* %[[BITCAST_PTR]], align 1

void check__writex18qword(unsigned long offset, unsigned __int64 data) {
  __writex18qword(offset, data);
}

// CHECK-MSVC: %[[DATA_ADDR:.*]] = alloca i64, align 8
// CHECK-MSVC: %[[OFFSET_ADDR:.*]] = alloca i32, align 4
// CHECK-MSVC: store i64 %data, i64* %[[DATA_ADDR]], align 8
// CHECK-MSVC: store i32 %offset, i32* %[[OFFSET_ADDR]], align 4
// CHECK-MSVC: %[[X18:.*]] = call i64 @llvm.read_register.i64(metadata ![[MD2]])
// CHECK-MSVC: %[[X18_AS_PTR:.*]] = inttoptr i64 %[[X18]] to i8*
// CHECK-MSVC: %[[OFFSET:.*]] = load i32, i32* %[[OFFSET_ADDR]], align 4
// CHECK-MSVC: %[[ZEXT_OFFSET:.*]] = zext i32 %[[OFFSET]] to i64
// CHECK-MSVC: %[[PTR:.*]] = getelementptr i8, i8* %[[X18_AS_PTR]], i64 %[[ZEXT_OFFSET]]
// CHECK-MSVC: %[[BITCAST_PTR:.*]] = bitcast i8* %[[PTR]] to i64*
// CHECK-MSVC: %[[DATA:.*]] = load i64, i64* %[[DATA_ADDR]], align 8
// CHECK-MSVC: store i64 %[[DATA]], i64* %[[BITCAST_PTR]], align 1

unsigned char check__readx18byte(unsigned long offset) {
  return __readx18byte(offset);
}

// CHECK-MSVC: %[[OFFSET_ADDR:.*]] = alloca i32, align 4
// CHECK-MSVC: store i32 %offset, i32* %[[OFFSET_ADDR]], align 4
// CHECK-MSVC: %[[X18:.*]] = call i64 @llvm.read_register.i64(metadata ![[MD2]])
// CHECK-MSVC: %[[X18_AS_PTR:.*]] = inttoptr i64 %[[X18]] to i8*
// CHECK-MSVC: %[[OFFSET:.*]] = load i32, i32* %[[OFFSET_ADDR]], align 4
// CHECK-MSVC: %[[ZEXT_OFFSET:.*]] = zext i32 %[[OFFSET]] to i64
// CHECK-MSVC: %[[PTR:.*]] = getelementptr i8, i8* %[[X18_AS_PTR]], i64 %[[ZEXT_OFFSET]]
// CHECK-MSVC: %[[RETVAL:.*]] = load i8, i8* %[[PTR]], align 1
// CHECK-MSVC: ret i8 %[[RETVAL]]

unsigned short check__readx18word(unsigned long offset) {
  return __readx18word(offset);
}

// CHECK-MSVC: %[[OFFSET_ADDR:.*]] = alloca i32, align 4
// CHECK-MSVC: store i32 %offset, i32* %[[OFFSET_ADDR]], align 4
// CHECK-MSVC: %[[X18:.*]] = call i64 @llvm.read_register.i64(metadata ![[MD2]])
// CHECK-MSVC: %[[X18_AS_PTR:.*]] = inttoptr i64 %[[X18]] to i8*
// CHECK-MSVC: %[[OFFSET:.*]] = load i32, i32* %[[OFFSET_ADDR]], align 4
// CHECK-MSVC: %[[ZEXT_OFFSET:.*]] = zext i32 %[[OFFSET]] to i64
// CHECK-MSVC: %[[PTR:.*]] = getelementptr i8, i8* %[[X18_AS_PTR]], i64 %[[ZEXT_OFFSET]]
// CHECK-MSVC: %[[BITCAST_PTR:.*]] = bitcast i8* %[[PTR]] to i16*
// CHECK-MSVC: %[[RETVAL:.*]] = load i16, i16* %[[BITCAST_PTR]], align 1
// CHECK-MSVC: ret i16 %[[RETVAL]]

unsigned long check__readx18dword(unsigned long offset) {
  return __readx18dword(offset);
}

// CHECK-MSVC: %[[OFFSET_ADDR:.*]] = alloca i32, align 4
// CHECK-MSVC: store i32 %offset, i32* %[[OFFSET_ADDR]], align 4
// CHECK-MSVC: %[[X18:.*]] = call i64 @llvm.read_register.i64(metadata ![[MD2]])
// CHECK-MSVC: %[[X18_AS_PTR:.*]] = inttoptr i64 %[[X18]] to i8*
// CHECK-MSVC: %[[OFFSET:.*]] = load i32, i32* %[[OFFSET_ADDR]], align 4
// CHECK-MSVC: %[[ZEXT_OFFSET:.*]] = zext i32 %[[OFFSET]] to i64
// CHECK-MSVC: %[[PTR:.*]] = getelementptr i8, i8* %[[X18_AS_PTR]], i64 %[[ZEXT_OFFSET]]
// CHECK-MSVC: %[[BITCAST_PTR:.*]] = bitcast i8* %[[PTR]] to i32*
// CHECK-MSVC: %[[RETVAL:.*]] = load i32, i32* %[[BITCAST_PTR]], align 1
// CHECK-MSVC: ret i32 %[[RETVAL]]

unsigned __int64 check__readx18qword(unsigned long offset) {
  return __readx18qword(offset);
}

// CHECK-MSVC: %[[OFFSET_ADDR:.*]] = alloca i32, align 4
// CHECK-MSVC: store i32 %offset, i32* %[[OFFSET_ADDR]], align 4
// CHECK-MSVC: %[[X18:.*]] = call i64 @llvm.read_register.i64(metadata ![[MD2]])
// CHECK-MSVC: %[[X18_AS_PTR:.*]] = inttoptr i64 %[[X18]] to i8*
// CHECK-MSVC: %[[OFFSET:.*]] = load i32, i32* %[[OFFSET_ADDR]], align 4
// CHECK-MSVC: %[[ZEXT_OFFSET:.*]] = zext i32 %[[OFFSET]] to i64
// CHECK-MSVC: %[[PTR:.*]] = getelementptr i8, i8* %[[X18_AS_PTR]], i64 %[[ZEXT_OFFSET]]
// CHECK-MSVC: %[[BITCAST_PTR:.*]] = bitcast i8* %[[PTR]] to i64*
// CHECK-MSVC: %[[RETVAL:.*]] = load i64, i64* %[[BITCAST_PTR]], align 1
// CHECK-MSVC: ret i64 %[[RETVAL]]

// CHECK-MSVC: ![[MD2]] = !{!"x18"}
// CHECK-MSVC: ![[MD3]] = !{!"sp"}
