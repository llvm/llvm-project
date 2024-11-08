// RUN: %clang_cc1 -triple arm64-windows -Wno-implicit-function-declaration -fms-compatibility -emit-llvm -o - %s \
// RUN:    | FileCheck %s --check-prefix=CHECK-MSVC --check-prefix=CHECK-MSCOMPAT

// RUN: not %clang_cc1 -triple arm64-linux -Werror -S -o /dev/null %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-LINUX

// RUN: %clang_cc1 -triple arm64-darwin -Wno-implicit-function-declaration -fms-compatibility -emit-llvm -o - %s \
// RUN:    | FileCheck %s -check-prefix CHECK-MSCOMPAT

long test_InterlockedAdd(long volatile *Addend, long Value) {
  return _InterlockedAdd(Addend, Value);
}

long test_InterlockedAdd_constant(long volatile *Addend) {
  return _InterlockedAdd(Addend, -1);
}

// CHECK-LABEL: define {{.*}} i32 @test_InterlockedAdd(ptr %Addend, i32 %Value) {{.*}} {
// CHECK-MSVC: %[[OLDVAL:[0-9]+]] = atomicrmw add ptr %1, i32 %2 seq_cst, align 4
// CHECK-MSVC: %[[NEWVAL:[0-9]+]] = add i32 %[[OLDVAL:[0-9]+]], %2
// CHECK-MSVC: ret i32 %[[NEWVAL:[0-9]+]]
// CHECK-LINUX: error: call to undeclared function '_InterlockedAdd'

__int64 test_InterlockedAdd64(__int64 volatile *Addend, __int64 Value) {
  return _InterlockedAdd64(Addend, Value);
}

__int64 test_InterlockedAdd64_constant(__int64 volatile *Addend) {
  return _InterlockedAdd64(Addend, -1);
}

// CHECK-LABEL: define {{.*}} i64 @test_InterlockedAdd64(ptr %Addend, i64 %Value) {{.*}} {
// CHECK-MSVC: %[[OLDVAL:[0-9]+]] = atomicrmw add ptr %1, i64 %2 seq_cst, align 8
// CHECK-MSVC: %[[NEWVAL:[0-9]+]] = add i64 %[[OLDVAL:[0-9]+]], %2
// CHECK-MSVC: ret i64 %[[NEWVAL:[0-9]+]]
// CHECK-LINUX: error: call to undeclared function '_InterlockedAdd64'

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

// CHECK-MSCOMPAT: call i64 @llvm.read_register.i64(metadata ![[MD2:.*]])
// CHECK-MSCOMPAT: call i64 @llvm.read_register.i64(metadata ![[MD3:.*]])

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
// CHECK-MSCOMPAT: %[[X18:.*]] = call i64 @llvm.read_register.i64(metadata ![[MD2]])
// CHECK-MSCOMPAT: %[[X18_AS_PTR:.*]] = inttoptr i64 %[[X18]] to ptr
// CHECK-MSCOMPAT: %[[OFFSET:.*]] = load i32, ptr %[[OFFSET_ADDR]], align 4
// CHECK-MSCOMPAT: %[[ZEXT_OFFSET:.*]] = zext i32 %[[OFFSET]] to i64
// CHECK-MSCOMPAT: %[[PTR:.*]] = getelementptr i8, ptr %[[X18_AS_PTR]], i64 %[[ZEXT_OFFSET]]
// CHECK-MSCOMPAT: %[[DATA:.*]] = load i8, ptr %[[DATA_ADDR]], align 1
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
// CHECK-MSCOMPAT: %[[X18:.*]] = call i64 @llvm.read_register.i64(metadata ![[MD2]])
// CHECK-MSCOMPAT: %[[X18_AS_PTR:.*]] = inttoptr i64 %[[X18]] to ptr
// CHECK-MSCOMPAT: %[[OFFSET:.*]] = load i32, ptr %[[OFFSET_ADDR]], align 4
// CHECK-MSCOMPAT: %[[ZEXT_OFFSET:.*]] = zext i32 %[[OFFSET]] to i64
// CHECK-MSCOMPAT: %[[PTR:.*]] = getelementptr i8, ptr %[[X18_AS_PTR]], i64 %[[ZEXT_OFFSET]]
// CHECK-MSCOMPAT: %[[DATA:.*]] = load i16, ptr %[[DATA_ADDR]], align 2
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
// CHECK-MSCOMPAT: %[[X18:.*]] = call i64 @llvm.read_register.i64(metadata ![[MD2]])
// CHECK-MSCOMPAT: %[[X18_AS_PTR:.*]] = inttoptr i64 %[[X18]] to ptr
// CHECK-MSCOMPAT: %[[OFFSET:.*]] = load i32, ptr %[[OFFSET_ADDR]], align 4
// CHECK-MSCOMPAT: %[[ZEXT_OFFSET:.*]] = zext i32 %[[OFFSET]] to i64
// CHECK-MSCOMPAT: %[[PTR:.*]] = getelementptr i8, ptr %[[X18_AS_PTR]], i64 %[[ZEXT_OFFSET]]
// CHECK-MSCOMPAT: %[[DATA:.*]] = load i32, ptr %[[DATA_ADDR]], align 4
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
// CHECK-MSCOMPAT: %[[X18:.*]] = call i64 @llvm.read_register.i64(metadata ![[MD2]])
// CHECK-MSCOMPAT: %[[X18_AS_PTR:.*]] = inttoptr i64 %[[X18]] to ptr
// CHECK-MSCOMPAT: %[[OFFSET:.*]] = load i32, ptr %[[OFFSET_ADDR]], align 4
// CHECK-MSCOMPAT: %[[ZEXT_OFFSET:.*]] = zext i32 %[[OFFSET]] to i64
// CHECK-MSCOMPAT: %[[PTR:.*]] = getelementptr i8, ptr %[[X18_AS_PTR]], i64 %[[ZEXT_OFFSET]]
// CHECK-MSCOMPAT: %[[DATA:.*]] = load i64, ptr %[[DATA_ADDR]], align 8
// CHECK-MSCOMPAT: store i64 %[[DATA]], ptr %[[PTR]], align 1

unsigned char check__readx18byte(unsigned LONG offset) {
  return __readx18byte(offset);
}

// CHECK-MSCOMPAT: %[[OFFSET_ADDR:.*]] = alloca i32, align 4
// CHECK-MSCOMPAT: store i32 %offset, ptr %[[OFFSET_ADDR]], align 4
// CHECK-MSCOMPAT: %[[X18:.*]] = call i64 @llvm.read_register.i64(metadata ![[MD2]])
// CHECK-MSCOMPAT: %[[X18_AS_PTR:.*]] = inttoptr i64 %[[X18]] to ptr
// CHECK-MSCOMPAT: %[[OFFSET:.*]] = load i32, ptr %[[OFFSET_ADDR]], align 4
// CHECK-MSCOMPAT: %[[ZEXT_OFFSET:.*]] = zext i32 %[[OFFSET]] to i64
// CHECK-MSCOMPAT: %[[PTR:.*]] = getelementptr i8, ptr %[[X18_AS_PTR]], i64 %[[ZEXT_OFFSET]]
// CHECK-MSCOMPAT: %[[RETVAL:.*]] = load i8, ptr %[[PTR]], align 1
// CHECK-MSCOMPAT: ret i8 %[[RETVAL]]

unsigned short check__readx18word(unsigned LONG offset) {
  return __readx18word(offset);
}

// CHECK-MSCOMPAT: %[[OFFSET_ADDR:.*]] = alloca i32, align 4
// CHECK-MSCOMPAT: store i32 %offset, ptr %[[OFFSET_ADDR]], align 4
// CHECK-MSCOMPAT: %[[X18:.*]] = call i64 @llvm.read_register.i64(metadata ![[MD2]])
// CHECK-MSCOMPAT: %[[X18_AS_PTR:.*]] = inttoptr i64 %[[X18]] to ptr
// CHECK-MSCOMPAT: %[[OFFSET:.*]] = load i32, ptr %[[OFFSET_ADDR]], align 4
// CHECK-MSCOMPAT: %[[ZEXT_OFFSET:.*]] = zext i32 %[[OFFSET]] to i64
// CHECK-MSCOMPAT: %[[PTR:.*]] = getelementptr i8, ptr %[[X18_AS_PTR]], i64 %[[ZEXT_OFFSET]]
// CHECK-MSCOMPAT: %[[RETVAL:.*]] = load i16, ptr %[[PTR]], align 1
// CHECK-MSCOMPAT: ret i16 %[[RETVAL]]

unsigned LONG check__readx18dword(unsigned LONG offset) {
  return __readx18dword(offset);
}

// CHECK-MSCOMPAT: %[[OFFSET_ADDR:.*]] = alloca i32, align 4
// CHECK-MSCOMPAT: store i32 %offset, ptr %[[OFFSET_ADDR]], align 4
// CHECK-MSCOMPAT: %[[X18:.*]] = call i64 @llvm.read_register.i64(metadata ![[MD2]])
// CHECK-MSCOMPAT: %[[X18_AS_PTR:.*]] = inttoptr i64 %[[X18]] to ptr
// CHECK-MSCOMPAT: %[[OFFSET:.*]] = load i32, ptr %[[OFFSET_ADDR]], align 4
// CHECK-MSCOMPAT: %[[ZEXT_OFFSET:.*]] = zext i32 %[[OFFSET]] to i64
// CHECK-MSCOMPAT: %[[PTR:.*]] = getelementptr i8, ptr %[[X18_AS_PTR]], i64 %[[ZEXT_OFFSET]]
// CHECK-MSCOMPAT: %[[RETVAL:.*]] = load i32, ptr %[[PTR]], align 1
// CHECK-MSCOMPAT: ret i32 %[[RETVAL]]

unsigned __int64 check__readx18qword(unsigned LONG offset) {
  return __readx18qword(offset);
}

// CHECK-MSCOMPAT: %[[OFFSET_ADDR:.*]] = alloca i32, align 4
// CHECK-MSCOMPAT: store i32 %offset, ptr %[[OFFSET_ADDR]], align 4
// CHECK-MSCOMPAT: %[[X18:.*]] = call i64 @llvm.read_register.i64(metadata ![[MD2]])
// CHECK-MSCOMPAT: %[[X18_AS_PTR:.*]] = inttoptr i64 %[[X18]] to ptr
// CHECK-MSCOMPAT: %[[OFFSET:.*]] = load i32, ptr %[[OFFSET_ADDR]], align 4
// CHECK-MSCOMPAT: %[[ZEXT_OFFSET:.*]] = zext i32 %[[OFFSET]] to i64
// CHECK-MSCOMPAT: %[[PTR:.*]] = getelementptr i8, ptr %[[X18_AS_PTR]], i64 %[[ZEXT_OFFSET]]
// CHECK-MSCOMPAT: %[[RETVAL:.*]] = load i64, ptr %[[PTR]], align 1
// CHECK-MSCOMPAT: ret i64 %[[RETVAL]]

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

void check__prefetch(void *arg1) {
  return __prefetch(arg1);
}

// CHECK-MSCOMPAT: %[[ARG1:.*]].addr = alloca ptr, align 8
// CHECK-MSCOMPAT: store ptr %[[ARG1]], ptr %[[ARG1]].addr, align 8
// CHECK-MSCOMPAT: %[[VAR0:.*]] = load ptr, ptr %[[ARG1]].addr, align 8
// CHECK-MSCOMPAT: call void @llvm.prefetch.p0(ptr %[[VAR0]], i32 0, i32 3, i32 1)
// CHECK-MSCOMPAT: ret void


// CHECK-MSCOMPAT: ![[MD2]] = !{!"x18"}
// CHECK-MSCOMPAT: ![[MD3]] = !{!"sp"}
