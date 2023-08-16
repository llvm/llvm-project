// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s --check-prefixes=64BIT --check-prefix=BOTH
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu \
// RUN:   -emit-llvm %s -o - -target-cpu pwr8 | FileCheck %s --check-prefixes=64BIT --check-prefix=BOTH
// RUN: %clang_cc1 -triple powerpc-unknown-aix \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s --check-prefixes=32BIT --check-prefix=BOTH
// RUN: %clang_cc1 -triple powerpc64-unknown-aix \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s --check-prefixes=64BIT --check-prefix=BOTH

// Will not be adding include files to avoid any dependencies on the system.
// Required for size_t. Usually found in stddef.h.
typedef __SIZE_TYPE__ size_t;

// BOTH-LABEL: @testabs(
// BOTH-NEXT:  entry:
// BOTH-NEXT:    [[A_ADDR:%.*]] = alloca i32, align 4
// BOTH-NEXT:    store i32 [[A:%.*]], ptr [[A_ADDR]], align 4
// BOTH-NEXT:    [[TMP0:%.*]] = load i32, ptr [[A_ADDR]], align 4
// BOTH-NEXT:    [[NEG:%.*]] = sub nsw i32 0, [[TMP0]]
// BOTH-NEXT:    [[ABSCOND:%.*]] = icmp slt i32 [[TMP0]], 0
// BOTH-NEXT:    [[ABS:%.*]] = select i1 [[ABSCOND]], i32 [[NEG]], i32 [[TMP0]]
// BOTH-NEXT:    ret i32 [[ABS]]
signed int testabs(signed int a) {
  return __abs(a);
}

// 64BIT-LABEL: @testlabs(
// 64BIT-NEXT:  entry:
// 64BIT-NEXT:    [[A_ADDR:%.*]] = alloca i64, align 8
// 64BIT-NEXT:    store i64 [[A:%.*]], ptr [[A_ADDR]], align 8
// 64BIT-NEXT:    [[TMP0:%.*]] = load i64, ptr [[A_ADDR]], align 8
// 64BIT-NEXT:    [[NEG:%.*]] = sub nsw i64 0, [[TMP0]]
// 64BIT-NEXT:    [[ABSCOND:%.*]] = icmp slt i64 [[TMP0]], 0
// 64BIT-NEXT:    [[ABS:%.*]] = select i1 [[ABSCOND]], i64 [[NEG]], i64 [[TMP0]]
// 64BIT-NEXT:    ret i64 [[ABS]]
//
// 32BIT-LABEL: @testlabs(
// 32BIT-NEXT:  entry:
// 32BIT-NEXT:    [[A_ADDR:%.*]] = alloca i32, align 4
// 32BIT-NEXT:    store i32 [[A:%.*]], ptr [[A_ADDR]], align 4
// 32BIT-NEXT:    [[TMP0:%.*]] = load i32, ptr [[A_ADDR]], align 4
// 32BIT-NEXT:    [[NEG:%.*]] = sub nsw i32 0, [[TMP0]]
// 32BIT-NEXT:    [[ABSCOND:%.*]] = icmp slt i32 [[TMP0]], 0
// 32BIT-NEXT:    [[ABS:%.*]] = select i1 [[ABSCOND]], i32 [[NEG]], i32 [[TMP0]]
// 32BIT-NEXT:    ret i32 [[ABS]]
//
signed long testlabs(signed long a) {
  return __labs(a);
}

// 64BIT-LABEL: @testllabs(
// 64BIT-NEXT:  entry:
// 64BIT-NEXT:    [[A_ADDR:%.*]] = alloca i64, align 8
// 64BIT-NEXT:    store i64 [[A:%.*]], ptr [[A_ADDR]], align 8
// 64BIT-NEXT:    [[TMP0:%.*]] = load i64, ptr [[A_ADDR]], align 8
// 64BIT-NEXT:    [[NEG:%.*]] = sub nsw i64 0, [[TMP0]]
// 64BIT-NEXT:    [[ABSCOND:%.*]] = icmp slt i64 [[TMP0]], 0
// 64BIT-NEXT:    [[ABS:%.*]] = select i1 [[ABSCOND]], i64 [[NEG]], i64 [[TMP0]]
// 64BIT-NEXT:    ret i64 [[ABS]]
//
// 32BIT-LABEL: @testllabs(
// 32BIT-NEXT:  entry:
// 32BIT-NEXT:    [[A_ADDR:%.*]] = alloca i64, align 8
// 32BIT-NEXT:    store i64 [[A:%.*]], ptr [[A_ADDR]], align 8
// 32BIT-NEXT:    [[TMP0:%.*]] = load i64, ptr [[A_ADDR]], align 8
// 32BIT-NEXT:    [[NEG:%.*]] = sub nsw i64 0, [[TMP0]]
// 32BIT-NEXT:    [[ABSCOND:%.*]] = icmp slt i64 [[TMP0]], 0
// 32BIT-NEXT:    [[ABS:%.*]] = select i1 [[ABSCOND]], i64 [[NEG]], i64 [[TMP0]]
// 32BIT-NEXT:    ret i64 [[ABS]]
//
signed long long testllabs(signed long long a) {
  return __llabs(a);
}

// 64BIT-LABEL: @testalloca(
// 64BIT:         [[TMP1:%.*]] = alloca i8, i64
// 64BIT-NEXT:    ret ptr [[TMP1]]
//
// 32BIT-LABEL: @testalloca(
// 32BIT:         [[TMP1:%.*]] = alloca i8, i32
// 32BIT-NEXT:    ret ptr [[TMP1]]
//
void *testalloca(size_t size) {
  return __alloca(size);
}

// Note that bpermd is 64 bit only.
#ifdef __PPC64__
// 64BIT-LABEL: @testbpermd(
// 64BIT:         [[TMP:%.*]] = call i64 @llvm.ppc.bpermd(i64 {{%.*}}, i64 {{%.*}})
// 64BIT-NEXT:    ret i64 [[TMP]]
//
long long testbpermd(long long bit_selector, long long source) {
  return __bpermd(bit_selector, source);
}
#endif

#ifdef __PPC64__
// 64BIT-LABEL: @testdivde(
// 64BIT:         [[TMP2:%.*]] = call i64 @llvm.ppc.divde
// 64BIT-NEXT:    ret i64 [[TMP2]]
long long testdivde(long long dividend, long long divisor) {
  return __divde(dividend, divisor);
}

// 64BIT-LABEL: @testdivdeu(
// 64BIT:         [[TMP2:%.*]] = call i64 @llvm.ppc.divdeu
// 64BIT-NEXT:    ret i64 [[TMP2]]
unsigned long long testdivdeu(unsigned long long dividend, unsigned long long divisor) {
  return __divdeu(dividend, divisor);
}
#endif

// 64BIT-LABEL: @testdivwe(
// 64BIT:         [[TMP2:%.*]] = call i32 @llvm.ppc.divwe
// 64BIT-NEXT:    ret i32 [[TMP2]]
//
// 32BIT-LABEL: @testdivwe(
// 32BIT:         [[TMP2:%.*]] = call i32 @llvm.ppc.divwe
// 32BIT-NEXT:    ret i32 [[TMP2]]
int testdivwe(int dividend, int divisor) {
  return __divwe(dividend, divisor);
}

// 64BIT-LABEL: @testdivweu(
// 64BIT:         [[TMP2:%.*]] = call i32 @llvm.ppc.divweu
// 64BIT-NEXT:    ret i32 [[TMP2]]
//
// 32BIT-LABEL: @testdivweu(
// 32BIT:         [[TMP2:%.*]] = call i32 @llvm.ppc.divweu
// 32BIT-NEXT:    ret i32 [[TMP2]]
unsigned int testdivweu(unsigned int dividend, unsigned int divisor) {
  return __divweu(dividend, divisor);
}

// BOTH-LABEL: @testfmadd(
// BOTH:         [[TMP3:%.*]] = call double @llvm.fma.f64
// BOTH-NEXT:    ret double [[TMP3]]
//
double testfmadd(double a, double b, double c) {
  return __fmadd(a, b, c);
}

// BOTH-LABEL: @testfmadds(
// BOTH:         [[TMP3:%.*]] = call float @llvm.fma.f32(
// BOTH-NEXT:    ret float [[TMP3]]
//
float testfmadds(float a, float b, float c) {
  return __fmadds(a, b, c);
}

// Required for bzero and bcopy. Usually in strings.h.
extern void bcopy(const void *__src, void *__dest, size_t __n);
extern void bzero(void *__s, size_t __n);

// 64BIT-LABEL: @testalignx(
// 64BIT:         call void @llvm.assume(i1 true) [ "align"(ptr {{%.*}}, i64 16) ]
// 64BIT-NEXT:    ret void
//
// 32BIT-LABEL: @testalignx(
// 32BIT:         call void @llvm.assume(i1 true) [ "align"(ptr {{%.*}}, i32 16) ]
// 32BIT-NEXT:    ret void
//
void testalignx(const void *pointer) {
  __alignx(16, pointer);
}

// 64BIT-LABEL: @testbcopy(
// 64BIT:         call void @llvm.memmove.p0.p0.i64(ptr align 1 {{%.*}}, ptr align 1 {{%.*}}, i64 {{%.*}}, i1 false)
// 64BIT-NEXT:    ret void
//
// 32BIT-LABEL: @testbcopy(
// 32BIT:         call void @llvm.memmove.p0.p0.i32(ptr align 1 {{%.*}}, ptr align 1 {{%.*}}, i32 {{%.*}}, i1 false)
// 32BIT-NEXT:    ret void
//
void testbcopy(const void *src, void *dest, size_t n) {
  bcopy(src, dest, n);
}

// 64BIT-LABEL: @testbzero(
// 64BIT:         call void @llvm.memset.p0.i64(ptr align 1 {{%.*}}, i8 0, i64 {{%.*}}, i1 false)
// 64BIT-NEXT:    ret void
//
// 32BIT-LABEL: @testbzero(
// 32BIT:         call void @llvm.memset.p0.i32(ptr align 1 {{%.*}}, i8 0, i32 {{%.*}}, i1 false)
// 32BIT-NEXT:    ret void
//
void testbzero(void *s, size_t n) {
  bzero(s, n);
}

// 64BIT-LABEL: @testdcbf(
// 64BIT:         call void @llvm.ppc.dcbf(ptr {{%.*}})
// 64BIT-NEXT:    ret void
//
// 32BIT-LABEL: @testdcbf(
// 32BIT:         call void @llvm.ppc.dcbf(ptr {{%.*}})
// 32BIT-NEXT:    ret void
//
void testdcbf(const void *addr) {
  __dcbf(addr);
}

// BOTH-LABEL: @testreadflm(
// BOTH:         [[TMP0:%.*]] = call double @llvm.ppc.readflm()
// BOTH-NEXT:    ret double [[TMP0]]
//
double testreadflm(void) {
  return __readflm();
}

// BOTH-LABEL: @testsetflm(
// BOTH:         [[TMP1:%.*]] = call double @llvm.ppc.setflm(double {{%.*}})
// BOTH-NEXT:    ret double [[TMP1]]
//
double testsetflm(double a) {
  return __setflm(a);
}

// BOTH-LABEL: @testsetrnd(
// BOTH:         [[TMP1:%.*]] = call double @llvm.ppc.setrnd(i32 {{%.*}})
// BOTH-NEXT:    ret double [[TMP1]]
//
double testsetrnd(int mode) {
  return __setrnd(mode);
}
