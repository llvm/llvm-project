// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ffixed-point -fclangir -Wno-unused-value -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ffixed-point -fclangir -Wno-unused-value -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ffixed-point -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

// Test basic fixed-point literals
void test_short_fract() {
  // CIR:  cir.func{{.*}} @test_short_fract
  // LLVM: void @test_short_fract
  // OGCG: void @test_short_fract
  short _Fract sf = 0.5hr;
  // CIR:  %{{.*}} = cir.const #cir.int<64> : !s8i
  // LLVM: store i8 64, ptr %{{.*}}, align 1
  // OGCG: store i8 64, ptr %{{.*}}, align 1
  unsigned short _Fract usf = 0.5uhr;
  // CIR:  %{{.*}} = cir.const #cir.int<128> : !u8i
  // LLVM: store i8 -128, ptr %{{.*}}, align 1
  // OGCG: store i8 -128, ptr %{{.*}}, align 1
}

void test_fract() {
  // CIR:  cir.func{{.*}} @test_fract
  // LLVM: void @test_fract
  // OGCG: void @test_fract
  _Fract f = 0.5r;
  // CIR:  %{{.*}} = cir.const #cir.int<16384> : !s16i
  // LLVM: store i16 16384, ptr %{{.*}}, align 2
  // OGCG: store i16 16384, ptr %{{.*}}, align 2
  unsigned _Fract uf = 0.5ur;
  // CIR:  %{{.*}} = cir.const #cir.int<32768> : !u16i
  // LLVM: store i16 -32768, ptr %{{.*}}, align 2
  // OGCG: store i16 -32768, ptr %{{.*}}, align 2
}

void test_long_fract() {
  // CIR:  cir.func{{.*}} @test_long_fract
  // LLVM: void @test_long_fract
  // OGCG: void @test_long_fract
  long _Fract lf = 0.5lr;
  // CIR:  %{{.*}} = cir.const #cir.int<1073741824> : !s32i
  // LLVM: store i32 1073741824, ptr %{{.*}}, align 4
  // OGCG: store i32 1073741824, ptr %{{.*}}, align 4
  unsigned long _Fract ulf = 0.5ulr;
  // CIR:  %{{.*}} = cir.const #cir.int<2147483648> : !u32i
  // LLVM: store i32 -2147483648, ptr %{{.*}}, align 4
  // OGCG: store i32 -2147483648, ptr %{{.*}}, align 4
}

void test_short_accum() {
  // CIR:  cir.func{{.*}} @test_short_accum
  // LLVM: void @test_short_accum
  // OGCG: void @test_short_accum
  short _Accum sa = 0.5hk;
  // CIR:  %{{.*}} = cir.const #cir.int<64> : !s16i
  // LLVM: store i16 64, ptr %{{.*}}, align 2
  // OGCG: store i16 64, ptr %{{.*}}, align 2
  unsigned short _Accum usa = 0.5uhk;
  // CIR:  %{{.*}} = cir.const #cir.int<128> : !u16i
  // LLVM: store i16 128, ptr %{{.*}}, align 2
  // OGCG: store i16 128, ptr %{{.*}}, align 2
}

void test_accum() {
  // CIR:  cir.func{{.*}} @test_accum
  // LLVM: void @test_accum
  // OGCG: void @test_accum
  _Accum a = 0.5k;
  // CIR:  %{{.*}} = cir.const #cir.int<16384> : !s32i
  // LLVM: store i32 16384, ptr %{{.*}}, align 4
  // OGCG: store i32 16384, ptr %{{.*}}, align 4
  unsigned _Accum ua = 0.5uk;
  // CIR:  %{{.*}} = cir.const #cir.int<32768> : !u32i
  // LLVM: store i32 32768, ptr %{{.*}}, align 4
  // OGCG: store i32 32768, ptr %{{.*}}, align 4
}

void test_long_accum() {
  // CIR:  cir.func{{.*}} @test_long_accum
  // LLVM: void @test_long_accum
  // OGCG: void @test_long_accum
  long _Accum la = 0.5lk;
  // CIR:  %{{.*}} = cir.const #cir.int<1073741824> : !s64i
  // LLVM: store i64 1073741824, ptr %{{.*}}, align 8
  // OGCG: store i64 1073741824, ptr %{{.*}}, align 8
  unsigned long _Accum ula = 0.5ulk;
  // CIR:  %{{.*}} = cir.const #cir.int<2147483648> : !u64i
  // LLVM: store i64 2147483648, ptr %{{.*}}, align 8
  // OGCG: store i64 2147483648, ptr %{{.*}}, align 8
}

void test_negative() {
  // CIR:  cir.func{{.*}} @test_negative
  // LLVM: void @test_negative
  // OGCG: void @test_negative
  short _Fract sf = -0.5hr;
  // CIR:  %{{.*}} = cir.const #cir.int<-64> : !s8i
  // LLVM: store i8 -64, ptr %{{.*}}, align 1
  // OGCG: store i8 -64, ptr %{{.*}}, align 1
  _Fract f = -0.5r;
  // CIR:  %{{.*}} = cir.const #cir.int<-16384> : !s16i
  // LLVM: store i16 -16384, ptr %{{.*}}, align 2
  // OGCG: store i16 -16384, ptr %{{.*}}, align 2
  long _Fract lf = -0.5lr;
  // CIR:  %{{.*}} = cir.const #cir.int<-1073741824> : !s32i
  // LLVM: store i32 -1073741824, ptr %{{.*}}, align 4
  // OGCG: store i32 -1073741824, ptr %{{.*}}, align 4
  short _Accum sa = -0.5hk;
  // CIR:  %{{.*}} = cir.const #cir.int<-64> : !s16i
  // LLVM: store i16 -64, ptr %{{.*}}, align 2
  // OGCG: store i16 -64, ptr %{{.*}}, align 2
  _Accum a = -0.5k;
  // CIR:  %{{.*}} = cir.const #cir.int<-16384> : !s32i
  // LLVM: store i32 -16384, ptr %{{.*}}, align 4
  // OGCG: store i32 -16384, ptr %{{.*}}, align 4
  long _Accum la = -0.5lk;
  // CIR:  %{{.*}} = cir.const #cir.int<-1073741824> : !s64i
  // LLVM: store i64 -1073741824, ptr %{{.*}}, align 8
  // OGCG: store i64 -1073741824, ptr %{{.*}}, align 8
}

// FIXME: `FixedPointCast` in CIR is not supported.
//        Only check valid for `_Sat` fixed point types,

void test_sat_short_accum() {
  // CIR:  cir.func{{.*}} @test_sat_short_accum
  // LLVM: void @test_sat_short_accum
  // OGCG: void @test_sat_short_accum
  _Sat short _Accum ssa;
  // CIR:  cir.alloca !s16i, !cir.ptr<!s16i>, ["ssa"]
  // LLVM: alloca i16, i64 1, align 2
  // OGCG: alloca i16, align 2
  _Sat unsigned short _Accum susa;
  // CIR:  cir.alloca !u16i, !cir.ptr<!u16i>, ["susa"]
  // LLVM: alloca i16, i64 1, align 2
  // OGCG: alloca i16, align 2
}

void test_sat_accum() {
  // CIR:  cir.func{{.*}} @test_sat_accum
  // LLVM: void @test_sat_accum
  // OGCG: void @test_sat_accum
  _Sat _Accum sa;
  // CIR:  cir.alloca !s32i, !cir.ptr<!s32i>, ["sa"]
  // LLVM: alloca i32, i64 1, align 4
  // OGCG: alloca i32, align 4
  _Sat unsigned _Accum sua;
  // CIR:  cir.alloca !u32i, !cir.ptr<!u32i>, ["sua"]
  // LLVM: alloca i32, i64 1, align 4
  // OGCG: alloca i32, align 4
}

void test_sat_long_accum() {
  // CIR:  cir.func{{.*}} @test_sat_long_accum
  // LLVM: void @test_sat_long_accum
  // OGCG: void @test_sat_long_accum
  _Sat long _Accum sla;
  // CIR:  cir.alloca !s64i, !cir.ptr<!s64i>, ["sla"]
  // LLVM: alloca i64, i64 1, align 8
  // OGCG: alloca i64, align 8
  _Sat unsigned long _Accum sula;
  // CIR:  cir.alloca !u64i, !cir.ptr<!u64i>, ["sula"]
  // LLVM: alloca i64, i64 1, align 8
  // OGCG: alloca i64, align 8
}

void test_sat_short_fract() {
  // CIR:  cir.func{{.*}} @test_sat_short_fract
  // LLVM: void @test_sat_short_fract
  // OGCG: void @test_sat_short_fract
  _Sat short _Fract ssf;
  // CIR:  cir.alloca !s8i, !cir.ptr<!s8i>, ["ssf"]
  // LLVM: alloca i8, i64 1, align 1
  // OGCG: alloca i8, align 1
  _Sat unsigned short _Fract susf;
  // CIR:  cir.alloca !u8i, !cir.ptr<!u8i>, ["susf"]
  // LLVM: alloca i8, i64 1, align 1
  // OGCG: alloca i8, align 1
}

void test_sat_fract() {
  // CIR:  cir.func{{.*}} @test_sat_fract
  // LLVM: void @test_sat_fract
  // OGCG: void @test_sat_fract
  _Sat _Fract sf;
  // CIR:  cir.alloca !s16i, !cir.ptr<!s16i>, ["sf"]
  // LLVM: alloca i16, i64 1, align 2
  // OGCG: alloca i16, align 2
  _Sat unsigned _Fract suf;
  // CIR:  cir.alloca !u16i, !cir.ptr<!u16i>, ["suf"]
  // LLVM: alloca i16, i64 1, align 2
  // OGCG: alloca i16, align 2
}

void test_sat_long_fract() {
  // CIR:  cir.func{{.*}} @test_sat_long_fract
  // LLVM: void @test_sat_long_fract
  // OGCG: void @test_sat_long_fract
  _Sat long _Fract slf;
  // CIR:  cir.alloca !s32i, !cir.ptr<!s32i>, ["slf"]
  // LLVM: alloca i32, i64 1, align 4
  // OGCG: alloca i32, align 4
  _Sat unsigned long _Fract sulf;
  // CIR:  cir.alloca !u32i, !cir.ptr<!u32i>, ["sulf"]
  // LLVM: alloca i32, i64 1, align 4
  // OGCG: alloca i32, align 4
}
