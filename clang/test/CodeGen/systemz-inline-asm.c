// RUN: %clang_cc1 -triple s390x-linux-gnu -O2 -emit-llvm -o - %s | FileCheck %s

unsigned int gi;
unsigned long gl;

void test_store_m(unsigned int i) {
  asm("st %1, %0" : "=m" (gi) : "r" (i));
// CHECK-LABEL: define void @test_store_m(i32 zeroext %i)
// CHECK: call void asm "st $1, $0", "=*m,r"(i32* nonnull @gi, i32 %i)
}

void test_store_Q(unsigned int i) {
  asm("st %1, %0" : "=Q" (gi) : "r" (i));
// CHECK-LABEL: define void @test_store_Q(i32 zeroext %i)
// CHECK: call void asm "st $1, $0", "=*Q,r"(i32* nonnull @gi, i32 %i)
}

void test_store_R(unsigned int i) {
  asm("st %1, %0" : "=R" (gi) : "r" (i));
// CHECK-LABEL: define void @test_store_R(i32 zeroext %i)
// CHECK: call void asm "st $1, $0", "=*R,r"(i32* nonnull @gi, i32 %i)
}

void test_store_S(unsigned int i) {
  asm("st %1, %0" : "=S" (gi) : "r" (i));
// CHECK-LABEL: define void @test_store_S(i32 zeroext %i)
// CHECK: call void asm "st $1, $0", "=*S,r"(i32* nonnull @gi, i32 %i)
}

void test_store_T(unsigned int i) {
  asm("st %1, %0" : "=T" (gi) : "r" (i));
// CHECK-LABEL: define void @test_store_T(i32 zeroext %i)
// CHECK: call void asm "st $1, $0", "=*T,r"(i32* nonnull @gi, i32 %i)
}

int test_load_m() {
  unsigned int i;
  asm("l %0, %1" : "=r" (i) : "m" (gi));
  return i;
// CHECK-LABEL: define signext i32 @test_load_m()
// CHECK: call i32 asm "l $0, $1", "=r,*m"(i32* nonnull @gi)
}

int test_load_Q() {
  unsigned int i;
  asm("l %0, %1" : "=r" (i) : "Q" (gi));
  return i;
// CHECK-LABEL: define signext i32 @test_load_Q()
// CHECK: call i32 asm "l $0, $1", "=r,*Q"(i32* nonnull @gi)
}

int test_load_R() {
  unsigned int i;
  asm("l %0, %1" : "=r" (i) : "R" (gi));
  return i;
// CHECK-LABEL: define signext i32 @test_load_R()
// CHECK: call i32 asm "l $0, $1", "=r,*R"(i32* nonnull @gi)
}

int test_load_S() {
  unsigned int i;
  asm("l %0, %1" : "=r" (i) : "S" (gi));
  return i;
// CHECK-LABEL: define signext i32 @test_load_S()
// CHECK: call i32 asm "l $0, $1", "=r,*S"(i32* nonnull @gi)
}

int test_load_T() {
  unsigned int i;
  asm("l %0, %1" : "=r" (i) : "T" (gi));
  return i;
// CHECK-LABEL: define signext i32 @test_load_T()
// CHECK: call i32 asm "l $0, $1", "=r,*T"(i32* nonnull @gi)
}

void test_mI(unsigned char *c) {
  asm volatile("cli %0, %1" :: "Q" (*c), "I" (100));
// CHECK-LABEL: define void @test_mI(i8* %c)
// CHECK: call void asm sideeffect "cli $0, $1", "*Q,I"(i8* %c, i32 100)
}

unsigned int test_dJa(unsigned int i, unsigned int j) {
  asm("sll %0, %2(%3)" : "=d" (i) : "0" (i), "J" (1000), "a" (j));
  return i;
// CHECK-LABEL: define zeroext i32 @test_dJa(i32 zeroext %i, i32 zeroext %j)
// CHECK: call i32 asm "sll $0, $2($3)", "=d,0,J,a"(i32 %i, i32 1000, i32 %j)
}

unsigned long test_rK(unsigned long i) {
  asm("aghi %0, %2" : "=r" (i) : "0" (i), "K" (-30000));
  return i;
// CHECK-LABEL: define i64 @test_rK(i64 %i)
// CHECK: call i64 asm "aghi $0, $2", "=r,0,K"(i64 %i, i32 -30000)
}

unsigned long test_rL(unsigned long i) {
  asm("sllg %0, %1, %2" : "=r" (i) : "r" (i), "L" (500000));
  return i;
// CHECK-LABEL: define i64 @test_rL(i64 %i)
// CHECK: call i64 asm "sllg $0, $1, $2", "=r,r,L"(i64 %i, i32 500000)
}

void test_M() {
  asm volatile("#FOO %0" :: "M"(0x7fffffff));
// CHECK-LABEL: define void @test_M()
// CHECK: call void asm sideeffect "#FOO $0", "M"(i32 2147483647)
}

float test_f32(float f, float g) {
  asm("aebr %0, %2" : "=f" (f) : "0" (f), "f" (g));
  return f;
// CHECK-LABEL: define float @test_f32(float %f, float %g)
// CHECK: call float asm "aebr $0, $2", "=f,0,f"(float %f, float %g)
}

double test_f64(double f, double g) {
  asm("adbr %0, %2" : "=f" (f) : "0" (f), "f" (g));
  return f;
// CHECK-LABEL: define double @test_f64(double %f, double %g)
// CHECK: call double asm "adbr $0, $2", "=f,0,f"(double %f, double %g)
}

long double test_f128(long double f, long double g) {
  asm("axbr %0, %2" : "=f" (f) : "0" (f), "f" (g));
  return f;
// CHECK: define void @test_f128(fp128* noalias nocapture sret(fp128) align 8 [[DEST:%.*]], fp128* nocapture readonly %0, fp128* nocapture readonly %1)
// CHECK: %f = load fp128, fp128* %0
// CHECK: %g = load fp128, fp128* %1
// CHECK: [[RESULT:%.*]] = tail call fp128 asm "axbr $0, $2", "=f,0,f"(fp128 %f, fp128 %g)
// CHECK: store fp128 [[RESULT]], fp128* [[DEST]]
}
