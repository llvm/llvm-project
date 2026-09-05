// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ffixed-point -fclangir -Wno-unused-value -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ffixed-point -fclangir -Wno-unused-value -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ffixed-point -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM

_Fract global_f = 0.5r;
// CIR: cir.global external @global_f = #cir.int<16384> : !s16i {alignment = 2 : i64}
// LLVM: @global_f = global i16 16384, align 2
_Accum global_a = 1.5k;
// CIR: cir.global external @global_a = #cir.int<49152> : !s32i {alignment = 4 : i64}
// LLVM: @global_a = global i32 49152, align 4

short _Fract global_short_f = 0.5hr;
// CIR: cir.global external @global_short_f = #cir.int<64> : !s8i {alignment = 1 : i64}
// LLVM: @global_short_f = global i8 64, align 1
short _Accum global_short_a = 1.5hk;
// CIR: cir.global external @global_short_a = #cir.int<192> : !s16i {alignment = 2 : i64}
// LLVM: @global_short_a = global i16 192, align 2

unsigned short _Fract global_unsigned_short_f = 0.5uhr;
// CIR: cir.global external @global_unsigned_short_f = #cir.int<128> : !u8i {alignment = 1 : i64}
// LLVM: @global_unsigned_short_f = global i8 -128, align 1

unsigned short _Accum global_unsigned_short_a = 1.5uhk;
// CIR: cir.global external @global_unsigned_short_a = #cir.int<384> : !u16i {alignment = 2 : i64}
// LLVM: @global_unsigned_short_a = global i16 384, align 2

// Test basic fixed-point literals
void test_short_fract() {
  // CIR:  cir.func{{.*}} @test_short_fract
  // LLVM: void @test_short_fract
  short _Fract sf = 0.5hr;
  // CIR:  %{{.*}} = cir.const #cir.int<64> : !s8i
  // LLVM: store i8 64, ptr %{{.*}}, align 1
  unsigned short _Fract usf = 0.5uhr;
  // CIR:  %{{.*}} = cir.const #cir.int<128> : !u8i
  // LLVM: store i8 -128, ptr %{{.*}}, align 1
}

void test_fract() {
  // CIR:  cir.func{{.*}} @test_fract
  // LLVM: void @test_fract
  _Fract f = 0.5r;
  // CIR:  %{{.*}} = cir.const #cir.int<16384> : !s16i
  // LLVM: store i16 16384, ptr %{{.*}}, align 2
  unsigned _Fract uf = 0.5ur;
  // CIR:  %{{.*}} = cir.const #cir.int<32768> : !u16i
  // LLVM: store i16 -32768, ptr %{{.*}}, align 2
}

void test_long_fract() {
  // CIR:  cir.func{{.*}} @test_long_fract
  // LLVM: void @test_long_fract
  long _Fract lf = 0.5lr;
  // CIR:  %{{.*}} = cir.const #cir.int<1073741824> : !s32i
  // LLVM: store i32 1073741824, ptr %{{.*}}, align 4
  unsigned long _Fract ulf = 0.5ulr;
  // CIR:  %{{.*}} = cir.const #cir.int<2147483648> : !u32i
  // LLVM: store i32 -2147483648, ptr %{{.*}}, align 4
}

void test_short_accum() {
  // CIR:  cir.func{{.*}} @test_short_accum
  // LLVM: void @test_short_accum
  short _Accum sa = 0.5hk;
  // CIR:  %{{.*}} = cir.const #cir.int<64> : !s16i
  // LLVM: store i16 64, ptr %{{.*}}, align 2
  unsigned short _Accum usa = 0.5uhk;
  // CIR:  %{{.*}} = cir.const #cir.int<128> : !u16i
  // LLVM: store i16 128, ptr %{{.*}}, align 2
}

void test_accum() {
  // CIR:  cir.func{{.*}} @test_accum
  // LLVM: void @test_accum
  _Accum a = 0.5k;
  // CIR:  %{{.*}} = cir.const #cir.int<16384> : !s32i
  // LLVM: store i32 16384, ptr %{{.*}}, align 4
  unsigned _Accum ua = 0.5uk;
  // CIR:  %{{.*}} = cir.const #cir.int<32768> : !u32i
  // LLVM: store i32 32768, ptr %{{.*}}, align 4
}

void test_long_accum() {
  // CIR:  cir.func{{.*}} @test_long_accum
  // LLVM: void @test_long_accum
  long _Accum la = 0.5lk;
  // CIR:  %{{.*}} = cir.const #cir.int<1073741824> : !s64i
  // LLVM: store i64 1073741824, ptr %{{.*}}, align 8
  unsigned long _Accum ula = 0.5ulk;
  // CIR:  %{{.*}} = cir.const #cir.int<2147483648> : !u64i
  // LLVM: store i64 2147483648, ptr %{{.*}}, align 8
}

void test_negative() {
  // CIR:  cir.func{{.*}} @test_negative
  // LLVM: void @test_negative
  short _Fract sf = -0.5hr;
  // CIR:  %{{.*}} = cir.const #cir.int<-64> : !s8i
  // LLVM: store i8 -64, ptr %{{.*}}, align 1
  _Fract f = -0.5r;
  // CIR:  %{{.*}} = cir.const #cir.int<-16384> : !s16i
  // LLVM: store i16 -16384, ptr %{{.*}}, align 2
  long _Fract lf = -0.5lr;
  // CIR:  %{{.*}} = cir.const #cir.int<-1073741824> : !s32i
  // LLVM: store i32 -1073741824, ptr %{{.*}}, align 4
  short _Accum sa = -0.5hk;
  // CIR:  %{{.*}} = cir.const #cir.int<-64> : !s16i
  // LLVM: store i16 -64, ptr %{{.*}}, align 2
  _Accum a = -0.5k;
  // CIR:  %{{.*}} = cir.const #cir.int<-16384> : !s32i
  // LLVM: store i32 -16384, ptr %{{.*}}, align 4
  long _Accum la = -0.5lk;
  // CIR:  %{{.*}} = cir.const #cir.int<-1073741824> : !s64i
  // LLVM: store i64 -1073741824, ptr %{{.*}}, align 8
}

// FIXME: `FixedPointCast` in CIR is not supported.
//        Only check valid for `_Sat` fixed point types,

void test_sat_short_accum() {
  // CIR:  cir.func{{.*}} @test_sat_short_accum
  // LLVM: void @test_sat_short_accum
  _Sat short _Accum ssa;
  // CIR:  cir.alloca "ssa" {{.*}} : !cir.ptr<!s16i>
  // LLVM: alloca i16, align 2
  _Sat unsigned short _Accum susa;
  // CIR:  cir.alloca "susa" {{.*}} : !cir.ptr<!u16i>
  // LLVM: alloca i16, align 2
}

void test_sat_accum() {
  // CIR:  cir.func{{.*}} @test_sat_accum
  // LLVM: void @test_sat_accum
  _Sat _Accum sa;
  // CIR:  cir.alloca "sa" {{.*}} : !cir.ptr<!s32i>
  // LLVM: alloca i32, align 4
  _Sat unsigned _Accum sua;
  // CIR:  cir.alloca "sua" {{.*}} : !cir.ptr<!u32i>
  // LLVM: alloca i32, align 4
}

void test_sat_long_accum() {
  // CIR:  cir.func{{.*}} @test_sat_long_accum
  // LLVM: void @test_sat_long_accum
  _Sat long _Accum sla;
  // CIR:  cir.alloca "sla" {{.*}} : !cir.ptr<!s64i>
  // LLVM: alloca i64, align 8
  _Sat unsigned long _Accum sula;
  // CIR:  cir.alloca "sula" {{.*}} : !cir.ptr<!u64i>
  // LLVM: alloca i64, align 8
}

void test_sat_short_fract() {
  // CIR:  cir.func{{.*}} @test_sat_short_fract
  // LLVM: void @test_sat_short_fract
  _Sat short _Fract ssf;
  // CIR:  cir.alloca "ssf" {{.*}} : !cir.ptr<!s8i>
  // LLVM: alloca i8, align 1
  _Sat unsigned short _Fract susf;
  // CIR:  cir.alloca "susf" {{.*}} : !cir.ptr<!u8i>
  // LLVM: alloca i8, align 1
}

void test_sat_fract() {
  // CIR:  cir.func{{.*}} @test_sat_fract
  // LLVM: void @test_sat_fract
  _Sat _Fract sf;
  // CIR:  cir.alloca "sf" {{.*}} : !cir.ptr<!s16i>
  // LLVM: alloca i16, align 2
  _Sat unsigned _Fract suf;
  // CIR:  cir.alloca "suf" {{.*}} : !cir.ptr<!u16i>
  // LLVM: alloca i16, align 2
}

void test_sat_long_fract() {
  // CIR:  cir.func{{.*}} @test_sat_long_fract
  // LLVM: void @test_sat_long_fract
  _Sat long _Fract slf;
  // CIR:  cir.alloca "slf" {{.*}} : !cir.ptr<!s32i>
  // LLVM: alloca i32, align 4
  _Sat unsigned long _Fract sulf;
  // CIR:  cir.alloca "sulf" {{.*}} : !cir.ptr<!u32i>
  // LLVM: alloca i32, align 4
}
