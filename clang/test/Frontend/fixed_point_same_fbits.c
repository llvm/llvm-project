// RUN: %clang -ffixed-point -S -emit-llvm -o - %s | FileCheck %s -check-prefix=DEFAULT
// RUN: %clang -ffixed-point -fsame-fbits -S -emit-llvm -o - %s | FileCheck %s -check-prefix=SAME

/* The scale for unsigned fixed point types should be the same as that of signed
 * fixed point types when -fsame-fbits is enabled. */

void func() {
  unsigned short _Accum u_short_accum = 0.5uhk;
  unsigned _Accum u_accum = 0.5uk;
  unsigned long _Accum u_long_accum = 0.5ulk;
  unsigned short _Fract u_short_fract = 0.5uhr;
  unsigned _Fract u_fract = 0.5ur;
  unsigned long _Fract u_long_fract = 0.5ulr;

// DEFAULT: store i16 128, i16* {{.*}}, align 2
// DEFAULT: store i32 32768, i32* {{.*}}, align 4
// DEFAULT: store i64 2147483648, i64* {{.*}}, align 8
// DEFAULT: store i8  -128, i8* {{.*}}, align 1
// DEFAULT: store i16 -32768, i16* {{.*}}, align 2
// DEFAULT: store i32 -2147483648, i32* {{.*}}, align 4

// SAME: store i16 64, i16* {{.*}}, align 2
// SAME: store i32 16384, i32* {{.*}}, align 4
// SAME: store i64 1073741824, i64* {{.*}}, align 8
// SAME: store i8  64, i8* {{.*}}, align 1
// SAME: store i16 16384, i16* {{.*}}, align 2
// SAME: store i32 1073741824, i32* {{.*}}, align 4
}
