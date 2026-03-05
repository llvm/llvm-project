// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ffixed-point -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// Test basic fixed-point literals
void test_short_fract() {
  // CHECK: cir.func{{.*}} @test_short_fract
  short _Fract sf = 0.5hr;
  // CHECK: %{{.*}} = cir.const #cir.int<64> : !s8i

  unsigned short _Fract usf = 0.5uhr;
  // CHECK: %{{.*}} = cir.const #cir.int<128> : !u8i
}

void test_fract() {
  // CHECK: cir.func{{.*}} @test_fract
  _Fract f = 0.5r;
  // CHECK: %{{.*}} = cir.const #cir.int<16384> : !s16i

  unsigned _Fract uf = 0.5ur;
  // CHECK: %{{.*}} = cir.const #cir.int<32768> : !u16i
}

void test_long_fract() {
  // CHECK: cir.func{{.*}} @test_long_fract
  long _Fract lf = 0.5lr;
  // CHECK: %{{.*}} = cir.const #cir.int<1073741824> : !s32i
}

void test_accum() {
  // CHECK: cir.func{{.*}} @test_accum
  short _Accum sa = 0.5hk;
  // CHECK: %{{.*}} = cir.const #cir.int<64> : !s16i
}

void test_negative() {
  // CHECK: cir.func{{.*}} @test_negative
  short _Fract sf = -0.5hr;
  // CHECK: %{{.*}} = cir.const #cir.int<64> : !s8i
  // CHECK: %{{.*}} = cir.unary(minus, %{{.*}}) : !s8i, !s8i
}
