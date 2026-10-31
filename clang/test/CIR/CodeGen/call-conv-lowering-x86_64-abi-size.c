// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// CIR-DAG: ![[U64U64:rec_anon_struct[0-9]*]] = !cir.struct<{data !u64i, data !u64i}>
// CIR-DAG: ![[S64U8:rec_anon_struct[0-9]*]] = !cir.struct<{data !s64i, data !u8i}>
// CIR-DAG: ![[U64S16:rec_anon_struct[0-9]*]] = !cir.struct<{data !u64i, data !s16i}>
// CIR-DAG: ![[S64S64:rec_anon_struct[0-9]*]] = !cir.struct<{data !s64i, data !s64i}>
// CIR-DAG: ![[U64S8:rec_anon_struct[0-9]*]] = !cir.struct<{data !u64i, data !s8i}>
// CIR-DAG: ![[S64U64:rec_anon_struct[0-9]*]] = !cir.struct<{data !s64i, data !u64i}>

// Each _BitInt(17) takes 4 bytes, so the last two fill the high eightbyte.
struct BI17x4 { _BitInt(17) a[4]; };
struct BI17x4 bi17x4(struct BI17x4 s) { return s; }

// CIR: cir.func {{.*}}@bi17x4(%arg0: !u64i loc({{[^)]+}}), %arg1: !u64i loc({{[^)]+}})) -> ![[U64U64]] attributes
// LLVM: define dso_local { i64, i64 } @bi17x4(i64 %{{.+}}, i64 %{{.+}})

struct IntBI17x3 { int x; _BitInt(17) a[3]; };
struct IntBI17x3 intbi17x3(struct IntBI17x3 s) { return s; }

// CIR: cir.func {{.*}}@intbi17x3(%arg0: !u64i loc({{[^)]+}}), %arg1: !u64i loc({{[^)]+}})) -> ![[U64U64]] attributes
// LLVM: define dso_local { i64, i64 } @intbi17x3(i64 %{{.+}}, i64 %{{.+}})

// The one-byte element is followed only by padding.
struct LongUBI3x1 { long l; unsigned _BitInt(3) a[1]; };
struct LongUBI3x1 longubi3x1(struct LongUBI3x1 s) { return s; }

// CIR: cir.func {{.*}}@longubi3x1(%arg0: !s64i loc({{[^)]+}}), %arg1: !u8i loc({{[^)]+}})) -> ![[S64U8]] attributes
// LLVM: define dso_local { i64, i8 } @longubi3x1(i64 %{{.+}}, i8 %{{.+}})

// Each _BitInt element is padded out to its alignment.
struct BI3x16 { _BitInt(3) a[16]; };
struct BI3x16 bi3x16(struct BI3x16 s) { return s; }

// CIR: cir.func {{.*}}@bi3x16(%arg0: !u64i loc({{[^)]+}}), %arg1: !u64i loc({{[^)]+}})) -> ![[U64U64]] attributes
// LLVM: define dso_local { i64, i64 } @bi3x16(i64 %{{.+}}, i64 %{{.+}})

struct BI9x5 { _BitInt(9) a[5]; };
struct BI9x5 bi9x5(struct BI9x5 s) { return s; }

// CIR: cir.func {{.*}}@bi9x5(%arg0: !u64i loc({{[^)]+}}), %arg1: !s16i loc({{[^)]+}})) -> ![[U64S16]] attributes
// LLVM: define dso_local { i64, i16 } @bi9x5(i64 %{{.+}}, i16 %{{.+}})

struct BI33x2 { _BitInt(33) a[2]; };
struct BI33x2 bi33x2(struct BI33x2 s) { return s; }

// CIR: cir.func {{.*}}@bi33x2(%arg0: !s64i loc({{[^)]+}}), %arg1: !s64i loc({{[^)]+}})) -> ![[S64S64]] attributes
// LLVM: define dso_local { i64, i64 } @bi33x2(i64 %{{.+}}, i64 %{{.+}})

struct CharBI7x8 { char c; _BitInt(7) a[8]; };
struct CharBI7x8 charbi7x8(struct CharBI7x8 s) { return s; }

// CIR: cir.func {{.*}}@charbi7x8(%arg0: !u64i loc({{[^)]+}}), %arg1: !s8i loc({{[^)]+}})) -> ![[U64S8]] attributes
// LLVM: define dso_local { i64, i8 } @charbi7x8(i64 %{{.+}}, i8 %{{.+}})

// The third element starts at byte 2, so the union's data reaches past the
// short.
union U3S { unsigned _BitInt(3) a[3]; short s; };
union U3S u3s(union U3S u) { return u; }

// CIR: cir.func {{.*}}@u3s(%arg0: !u32i loc({{[^)]+}})) -> !u32i attributes
// LLVM: define dso_local i32 @u3s(i32 %{{.+}})

// The ninth element shares the high eightbyte with the float.
struct UBI3x9Float { unsigned _BitInt(3) a[9]; float f; };
struct UBI3x9Float ubi3x9float(struct UBI3x9Float s) { return s; }

// CIR: cir.func {{.*}}@ubi3x9float(%arg0: !u64i loc({{[^)]+}}), %arg1: !u64i loc({{[^)]+}})) -> ![[U64U64]] attributes
// LLVM: define dso_local { i64, i64 } @ubi3x9float(i64 %{{.+}}, i64 %{{.+}})

// A narrow _BitInt bit-field counts at the size of its type, a whole byte, so
// it reaches the second byte.
struct __attribute__((aligned(2))) BitIntBitField {
  unsigned char x : 3;
  _BitInt(5) y : 2;
};
struct BitIntBitField bitintbitfield(struct BitIntBitField s) { return s; }

// CIR: cir.func {{.*}}@bitintbitfield(%arg0: !u16i loc({{[^)]+}})) -> !u16i attributes
// LLVM: define dso_local i16 @bitintbitfield(i16 %{{.+}})

// The long double covers all 16 bytes, so the high eightbyte is a whole i64 in
// either member order.
struct __attribute__((aligned(16))) LongShort { long a; short b; };
union LDFirst { long double ld; struct LongShort s; };
union LDFirst ldfirst(union LDFirst u) { return u; }

// CIR: cir.func {{.*}}@ldfirst(%arg0: !s64i loc({{[^)]+}}), %arg1: !u64i loc({{[^)]+}})) -> ![[S64U64]] attributes
// LLVM: define dso_local { i64, i64 } @ldfirst(i64 %{{.+}}, i64 %{{.+}})

union LSFirst { struct LongShort s; long double ld; };
union LSFirst lsfirst(union LSFirst u) { return u; }

// CIR: cir.func {{.*}}@lsfirst(%arg0: !s64i loc({{[^)]+}}), %arg1: !u64i loc({{[^)]+}})) -> ![[S64U64]] attributes
// LLVM: define dso_local { i64, i64 } @lsfirst(i64 %{{.+}}, i64 %{{.+}})

// On the stack a _BitInt is coerced to the integer covering its storage.
void stkbi17(long a, long b, long c, long d, long e, long f, _BitInt(17) x) {}

// CIR: cir.func {{.*}}@stkbi17(%arg0: !s64i {llvm.noundef} loc({{[^)]+}}), %arg1: !s64i {llvm.noundef} loc({{[^)]+}}), %arg2: !s64i {llvm.noundef} loc({{[^)]+}}), %arg3: !s64i {llvm.noundef} loc({{[^)]+}}), %arg4: !s64i {llvm.noundef} loc({{[^)]+}}), %arg5: !s64i {llvm.noundef} loc({{[^)]+}}), %arg6: !u32i {llvm.noundef} loc({{[^)]+}})) attributes
// LLVM: define dso_local void @stkbi17(i64 noundef %{{.+}}, i64 noundef %{{.+}}, i64 noundef %{{.+}}, i64 noundef %{{.+}}, i64 noundef %{{.+}}, i64 noundef %{{.+}}, i32 noundef %{{.+}})

void stkubi3(long a, long b, long c, long d, long e, long f,
             unsigned _BitInt(3) x) {}

// CIR: cir.func {{.*}}@stkubi3(%arg0: !s64i {llvm.noundef} loc({{[^)]+}}), %arg1: !s64i {llvm.noundef} loc({{[^)]+}}), %arg2: !s64i {llvm.noundef} loc({{[^)]+}}), %arg3: !s64i {llvm.noundef} loc({{[^)]+}}), %arg4: !s64i {llvm.noundef} loc({{[^)]+}}), %arg5: !s64i {llvm.noundef} loc({{[^)]+}}), %arg6: !u8i {llvm.noundef} loc({{[^)]+}})) attributes
// LLVM: define dso_local void @stkubi3(i64 noundef %{{.+}}, i64 noundef %{{.+}}, i64 noundef %{{.+}}, i64 noundef %{{.+}}, i64 noundef %{{.+}}, i64 noundef %{{.+}}, i8 noundef %{{.+}})

// A char or float vector is not a bool vector, so the eightbyte holding it
// stays an i64.
typedef char c1 __attribute__((ext_vector_type(1)));
struct LongC1 { long l; c1 v; };
struct LongC1 longc1(struct LongC1 s) { return s; }

// CIR: cir.func {{.*}}@longc1(%arg0: !s64i loc({{[^)]+}}), %arg1: !u64i loc({{[^)]+}})) -> ![[S64U64]] attributes
// LLVM: define dso_local { i64, i64 } @longc1(i64 %{{.+}}, i64 %{{.+}})

typedef float f1 __attribute__((ext_vector_type(1)));
struct LongF1 { long l; f1 v; };
struct LongF1 longf1(struct LongF1 s) { return s; }

// CIR: cir.func {{.*}}@longf1(%arg0: !s64i loc({{[^)]+}}), %arg1: !u64i loc({{[^)]+}})) -> ![[S64U64]] attributes
// LLVM: define dso_local { i64, i64 } @longf1(i64 %{{.+}}, i64 %{{.+}})
