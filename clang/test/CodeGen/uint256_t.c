// RUN: %clang_cc1 %s -emit-llvm -o - -triple=x86_64-apple-darwin9 | FileCheck %s

// Basic arithmetic code generation for __uint256_t / __int256_t.
// Verifies that all operations lower to i256 LLVM IR.
// On x86-64, __int256 is passed/returned via byval/sret (Memory class).

// CHECK-LABEL: define{{.*}} void @add256(ptr{{.*}}sret(i256)
// CHECK: add nsw i256
__int256_t add256(__int256_t a, __int256_t b) { return a + b; }

// CHECK-LABEL: define{{.*}} void @sub256(ptr{{.*}}sret(i256)
// CHECK: sub nsw i256
__int256_t sub256(__int256_t a, __int256_t b) { return a - b; }

// CHECK-LABEL: define{{.*}} void @mul256(ptr{{.*}}sret(i256)
// CHECK: mul i256
__uint256_t mul256(__uint256_t a, __uint256_t b) { return a * b; }

// CHECK-LABEL: define{{.*}} void @div256(ptr{{.*}}sret(i256)
// CHECK: udiv i256
__uint256_t div256(__uint256_t a, __uint256_t b) { return a / b; }

// CHECK-LABEL: define{{.*}} void @sdiv256(ptr{{.*}}sret(i256)
// CHECK: sdiv i256
__int256_t sdiv256(__int256_t a, __int256_t b) { return a / b; }

// Bitwise operations -- core of Hamming distance / popcount patterns
// CHECK-LABEL: define{{.*}} void @xor256(ptr{{.*}}sret(i256)
// CHECK: xor i256
__uint256_t xor256(__uint256_t a, __uint256_t b) { return a ^ b; }

// CHECK-LABEL: define{{.*}} void @and256(ptr{{.*}}sret(i256)
// CHECK: and i256
__uint256_t and256(__uint256_t a, __uint256_t b) { return a & b; }

// CHECK-LABEL: define{{.*}} void @or256(ptr{{.*}}sret(i256)
// CHECK: or i256
__uint256_t or256(__uint256_t a, __uint256_t b) { return a | b; }

// CHECK-LABEL: define{{.*}} void @not256(ptr{{.*}}sret(i256)
// CHECK: xor i256 %{{.*}}, -1
__uint256_t not256(__uint256_t a) { return ~a; }

// CHECK-LABEL: define{{.*}} void @shl256(ptr{{.*}}sret(i256)
// CHECK: shl i256
__uint256_t shl256(__uint256_t a, __uint256_t b) { return a << b; }

// CHECK-LABEL: define{{.*}} void @lshr256(ptr{{.*}}sret(i256)
// CHECK: lshr i256
__uint256_t lshr256(__uint256_t a, __uint256_t b) { return a >> b; }

// CHECK-LABEL: define{{.*}} void @ashr256(ptr{{.*}}sret(i256)
// CHECK: ashr i256
__int256_t ashr256(__int256_t a, __int256_t b) { return a >> b; }

// Widening conversion from uint64_t
// CHECK-LABEL: define{{.*}} void @widen(ptr{{.*}}sret(i256){{.*}}, i64
// CHECK: zext i64 %{{.*}} to i256
__uint256_t widen(unsigned long long x) { return (__uint256_t)x; }

// Narrowing conversion to uint64_t
// CHECK-LABEL: define{{.*}} i64 @narrow(ptr{{.*}}byval(i256)
// CHECK: trunc i256 %{{.*}} to i64
unsigned long long narrow(__uint256_t x) { return (unsigned long long)x; }

// Conversion between i128 and i256
// CHECK-LABEL: define{{.*}} void @from128(ptr{{.*}}sret(i256){{.*}}, i128
// CHECK: sext i128 %{{.*}} to i256
__int256_t from128(__int128_t x) { return (__int256_t)x; }

// CHECK-LABEL: define{{.*}} i128 @to128(ptr{{.*}}byval(i256)
// CHECK: trunc i256 %{{.*}} to i128
__int128_t to128(__int256_t x) { return (__int128_t)x; }

// Comparison
// CHECK-LABEL: define{{.*}} i32 @cmp_eq(ptr{{.*}}byval(i256){{.*}}, ptr{{.*}}byval(i256)
// CHECK: icmp eq i256
int cmp_eq(__int256_t a, __int256_t b) { return a == b; }

// CHECK-LABEL: define{{.*}} i32 @cmp_slt(ptr{{.*}}byval(i256){{.*}}, ptr{{.*}}byval(i256)
// CHECK: icmp slt i256
int cmp_slt(__int256_t a, __int256_t b) { return a < b; }

// CHECK-LABEL: define{{.*}} i32 @cmp_ult(ptr{{.*}}byval(i256){{.*}}, ptr{{.*}}byval(i256)
// CHECK: icmp ult i256
int cmp_ult(__uint256_t a, __uint256_t b) { return a < b; }

// Unsigned remainder
// CHECK-LABEL: define{{.*}} void @urem256(ptr{{.*}}sret(i256)
// CHECK: urem i256
__uint256_t urem256(__uint256_t a, __uint256_t b) { return a % b; }

// Signed remainder
// CHECK-LABEL: define{{.*}} void @srem256(ptr{{.*}}sret(i256)
// CHECK: srem i256
__int256_t srem256(__int256_t a, __int256_t b) { return a % b; }

// Unary minus
// CHECK-LABEL: define{{.*}} void @neg256(ptr{{.*}}sret(i256)
// CHECK: sub nsw i256 0,
__int256_t neg256(__int256_t a) { return -a; }

// Bool conversion
// CHECK-LABEL: define{{.*}} i32 @bool256(ptr{{.*}}byval(i256)
// CHECK: icmp ne i256 %{{.*}}, 0
int bool256(__uint256_t a) { return !!a; }

// ===----------------------------------------------------------------------===
// Comprehensive cast / conversion tests
// ===----------------------------------------------------------------------===

// --- Widening: signed small -> signed i256 (sign-extend) ---

// CHECK-LABEL: define{{.*}} void @widen_schar(ptr{{.*}}sret(i256)
// CHECK: sext i8 %{{.*}} to i256
__int256_t widen_schar(signed char x) { return (__int256_t)x; }

// CHECK-LABEL: define{{.*}} void @widen_short(ptr{{.*}}sret(i256)
// CHECK: sext i16 %{{.*}} to i256
__int256_t widen_short(short x) { return (__int256_t)x; }

// CHECK-LABEL: define{{.*}} void @widen_int(ptr{{.*}}sret(i256)
// CHECK: sext i32 %{{.*}} to i256
__int256_t widen_int(int x) { return (__int256_t)x; }

// CHECK-LABEL: define{{.*}} void @widen_long(ptr{{.*}}sret(i256)
// CHECK: sext i64 %{{.*}} to i256
__int256_t widen_long(long long x) { return (__int256_t)x; }

// --- Widening: unsigned small -> unsigned i256 (zero-extend) ---

// CHECK-LABEL: define{{.*}} void @widen_uchar(ptr{{.*}}sret(i256)
// CHECK: zext i8 %{{.*}} to i256
__uint256_t widen_uchar(unsigned char x) { return (__uint256_t)x; }

// CHECK-LABEL: define{{.*}} void @widen_ushort(ptr{{.*}}sret(i256)
// CHECK: zext i16 %{{.*}} to i256
__uint256_t widen_ushort(unsigned short x) { return (__uint256_t)x; }

// CHECK-LABEL: define{{.*}} void @widen_uint(ptr{{.*}}sret(i256)
// CHECK: zext i32 %{{.*}} to i256
__uint256_t widen_uint(unsigned int x) { return (__uint256_t)x; }

// CHECK-LABEL: define{{.*}} void @widen_ulong(ptr{{.*}}sret(i256)
// CHECK: zext i64 %{{.*}} to i256
__uint256_t widen_ulong(unsigned long long x) { return (__uint256_t)x; }

// --- Widening: unsigned i128 -> unsigned i256 (zero-extend) ---

// CHECK-LABEL: define{{.*}} void @widen_u128(ptr{{.*}}sret(i256)
// CHECK: zext i128 %{{.*}} to i256
__uint256_t widen_u128(__uint128_t x) { return (__uint256_t)x; }

// --- Widening: signed small -> unsigned i256 (sign-extend then implicit) ---
// C semantics: cast to __int256_t first (sext), then to __uint256_t (nop).
// The compiler folds this to sext directly to i256.

// CHECK-LABEL: define{{.*}} void @widen_schar_to_u256(ptr{{.*}}sret(i256)
// CHECK: sext i8 %{{.*}} to i256
__uint256_t widen_schar_to_u256(signed char x) { return (__uint256_t)x; }

// CHECK-LABEL: define{{.*}} void @widen_int_to_u256(ptr{{.*}}sret(i256)
// CHECK: sext i32 %{{.*}} to i256
__uint256_t widen_int_to_u256(int x) { return (__uint256_t)x; }

// --- Narrowing: i256 -> small types (truncate) ---

// CHECK-LABEL: define{{.*}} signext i8 @narrow_to_schar(ptr{{.*}}byval(i256)
// CHECK: trunc i256 %{{.*}} to i8
signed char narrow_to_schar(__int256_t x) { return (signed char)x; }

// CHECK-LABEL: define{{.*}} zeroext i8 @narrow_to_uchar(ptr{{.*}}byval(i256)
// CHECK: trunc i256 %{{.*}} to i8
unsigned char narrow_to_uchar(__uint256_t x) { return (unsigned char)x; }

// CHECK-LABEL: define{{.*}} signext i16 @narrow_to_short(ptr{{.*}}byval(i256)
// CHECK: trunc i256 %{{.*}} to i16
short narrow_to_short(__int256_t x) { return (short)x; }

// CHECK-LABEL: define{{.*}} zeroext i16 @narrow_to_ushort(ptr{{.*}}byval(i256)
// CHECK: trunc i256 %{{.*}} to i16
unsigned short narrow_to_ushort(__uint256_t x) { return (unsigned short)x; }

// CHECK-LABEL: define{{.*}} i32 @narrow_to_int(ptr{{.*}}byval(i256)
// CHECK: trunc i256 %{{.*}} to i32
int narrow_to_int(__int256_t x) { return (int)x; }

// CHECK-LABEL: define{{.*}} i32 @narrow_to_uint(ptr{{.*}}byval(i256)
// CHECK: trunc i256 %{{.*}} to i32
unsigned int narrow_to_uint(__uint256_t x) { return (unsigned int)x; }

// CHECK-LABEL: define{{.*}} i64 @narrow_to_long(ptr{{.*}}byval(i256)
// CHECK: trunc i256 %{{.*}} to i64
long long narrow_to_long(__int256_t x) { return (long long)x; }

// CHECK-LABEL: define{{.*}} i64 @narrow_to_ulong(ptr{{.*}}byval(i256)
// CHECK: trunc i256 %{{.*}} to i64
unsigned long long narrow_to_ulong(__uint256_t x) {
  return (unsigned long long)x;
}

// --- Narrowing: i256 -> i128 (unsigned) ---

// CHECK-LABEL: define{{.*}} i128 @narrow_to_u128(ptr{{.*}}byval(i256)
// CHECK: trunc i256 %{{.*}} to i128
__uint128_t narrow_to_u128(__uint256_t x) { return (__uint128_t)x; }

// --- Cross-sign: signed <-> unsigned i256 (no-op, same bit pattern) ---

// CHECK-LABEL: define{{.*}} void @signed_to_unsigned(ptr{{.*}}sret(i256){{.*}}, ptr{{.*}}byval(i256)
// CHECK-NOT: ext
// CHECK-NOT: trunc
// CHECK: ret void
__uint256_t signed_to_unsigned(__int256_t x) { return (__uint256_t)x; }

// CHECK-LABEL: define{{.*}} void @unsigned_to_signed(ptr{{.*}}sret(i256){{.*}}, ptr{{.*}}byval(i256)
// CHECK-NOT: ext
// CHECK-NOT: trunc
// CHECK: ret void
__int256_t unsigned_to_signed(__uint256_t x) { return (__int256_t)x; }

// --- Multi-step: negative char -> signed i256 (sign-extension across
// 248 bits) ---
// This verifies that (int256_t)(char)-42 produces a 256-bit -42
// via sign-extension, not a large positive number.

// CHECK-LABEL: define{{.*}} void @neg_char_to_i256(ptr{{.*}}sret(i256)
// CHECK: sext i8 %{{.*}} to i256
__int256_t neg_char_to_i256(signed char x) { return x; }

// --- Implicit conversions (no explicit cast) ---

// CHECK-LABEL: define{{.*}} void @implicit_int_to_i256(ptr{{.*}}sret(i256)
// CHECK: sext i32 %{{.*}} to i256
__int256_t implicit_int_to_i256(int x) { return x; }

// CHECK-LABEL: define{{.*}} i32 @implicit_i256_to_int(ptr{{.*}}byval(i256)
// CHECK: trunc i256 %{{.*}} to i32
int implicit_i256_to_int(__int256_t x) { return x; }
