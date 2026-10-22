// RUN: %check_clang_tidy %s bugprone-misplaced-widening-cast %t -- \
// RUN:     -config="{CheckOptions: {bugprone-misplaced-widening-cast.CheckImplicitCasts: true}}" \
// RUN:     -- -target x86_64-unknown-unknown
// RUN: %check_clang_tidy %s bugprone-misplaced-widening-cast %t -- \
// RUN:     -config="{CheckOptions: {bugprone-misplaced-widening-cast.CheckImplicitCasts: true}}" \
// RUN:     -- -target i386-unknown-unknown

// Tests rely on specific type sizes:
// unsigned int = 32, unsigned short = 16, unsigned char = 8,
// unsigned long = 64, unsigned long long = 64 bits.

struct BitfieldHeader {
  unsigned long long field32 : 32;
  unsigned long field16 : 16;
  unsigned int field8 : 8;
  unsigned long long field40 : 40;
  unsigned long long field24 : 24;
  long long sfield32 : 32;
  long sfield16 : 16;
};

// --- Implicit casts: no widening cases ---

// 32-bit bit field from unsigned int (32-bit) — no widening.
void bitfield32_shift(unsigned int size) {
  struct BitfieldHeader h = {};
  h.field32 = size << 1U;
}

void bitfield32_multiply(unsigned int size) {
  struct BitfieldHeader h = {};
  h.field32 = size * 2U;
}

void bitfield32_add(unsigned int size) {
  struct BitfieldHeader h = {};
  h.field32 = size + 1U;
}

void bitfield32_subtract(unsigned int size) {
  struct BitfieldHeader h = {};
  h.field32 = size - 1U;
}

void bitfield32_not(unsigned int size) {
  struct BitfieldHeader h = {};
  h.field32 = ~size;
}

// 16-bit bit field from unsigned short (16-bit) — no widening.
// Note: integer promotion makes CalcType 'int' (32-bit), but bit field is 16-bit,
// so this is truncation, not widening. No warning expected.
void bitfield16_shift(unsigned short size) {
  struct BitfieldHeader h = {};
  h.field16 = size << 1;
}

void bitfield16_multiply(unsigned short size) {
  struct BitfieldHeader h = {};
  h.field16 = size * 2;
}

void bitfield16_add(unsigned short size) {
  struct BitfieldHeader h = {};
  h.field16 = size + 1;
}

void bitfield16_subtract(unsigned short size) {
  struct BitfieldHeader h = {};
  h.field16 = size - 1;
}

void bitfield16_not(unsigned short size) {
  struct BitfieldHeader h = {};
  h.field16 = ~size;
}

// 8-bit bit field from unsigned char (8-bit) — no widening.
// Same: integer promotion makes CalcType 'int' (32-bit), bit field is 8-bit = truncation.
void bitfield8_shift(unsigned char size) {
  struct BitfieldHeader h = {};
  h.field8 = size << 1;
}

void bitfield8_multiply(unsigned char size) {
  struct BitfieldHeader h = {};
  h.field8 = size * 2;
}

void bitfield8_add(unsigned char size) {
  struct BitfieldHeader h = {};
  h.field8 = size + 1;
}

void bitfield8_subtract(unsigned char size) {
  struct BitfieldHeader h = {};
  h.field8 = size - 1;
}

void bitfield8_not(unsigned char size) {
  struct BitfieldHeader h = {};
  h.field8 = ~size;
}

// --- Implicit casts: widening cases (should warn) ---

// 40-bit bit field from unsigned int (32-bit) — widening DOES occur. Should warn.
void bitfield40_shift(unsigned int size) {
  struct BitfieldHeader h = {};
  h.field40 = size << 1U;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: either cast from 'unsigned int' to 'unsigned long long'
}

void bitfield40_multiply(unsigned int size) {
  struct BitfieldHeader h = {};
  h.field40 = size * 2U;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: either cast from 'unsigned int' to 'unsigned long long'
}

void bitfield40_add(unsigned int size) {
  struct BitfieldHeader h = {};
  h.field40 = size + 1U;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: either cast from 'unsigned int' to 'unsigned long long'
}

void bitfield40_subtract(unsigned int size) {
  struct BitfieldHeader h = {};
  h.field40 = size - 1U;
  // FIXME: checker doesn't detect potential widening for subtraction.
  // E.g. if size==0, result is 0xFFFFFFFF (32-bit), but in 40-bit space
  // it should be 0xFFFFFFFFFF. Limitation of getMaxCalculationWidth.
}

void bitfield40_not(unsigned int size) {
  struct BitfieldHeader h = {};
  h.field40 = ~size;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: either cast from 'unsigned int' to 'unsigned long long'
}

// --- Implicit casts: truncation cases (no warning) ---

// 24-bit bit field from unsigned short (16-bit) — after integer promotion,
// CalcType is 'int' (32-bit) which is wider than the 24-bit bit field.
// This is truncation, not widening. No warning expected.
void bitfield24_shift(unsigned short size) {
  struct BitfieldHeader h = {};
  h.field24 = size << 1;
}

void bitfield24_multiply(unsigned short size) {
  struct BitfieldHeader h = {};
  h.field24 = size * 2;
}

void bitfield24_add(unsigned short size) {
  struct BitfieldHeader h = {};
  h.field24 = size + 1;
}

void bitfield24_subtract(unsigned short size) {
  struct BitfieldHeader h = {};
  h.field24 = size - 1;
}

void bitfield24_not(unsigned short size) {
  struct BitfieldHeader h = {};
  h.field24 = ~size;
}

// --- Explicit casts with bit fields ---

// Source (unsigned short, 16-bit) == bit field width (16-bit). No warnings.
void explicit_cast_same_to_declared(unsigned short size) {
  struct BitfieldHeader h = {};
  h.field16 = (unsigned long)(size << 1);
}

void explicit_cast_same_to_bitfield(unsigned short size) {
  struct BitfieldHeader h = {};
  h.field16 = (unsigned short)(size << 1);
}

void explicit_cast_same_to_narrower(unsigned short size) {
  struct BitfieldHeader h = {};
  h.field16 = (unsigned char)(size << 1);
}

// Source (unsigned int, 32-bit) > bit field width (16-bit). Truncation, no warnings.
void explicit_cast_wider_to_declared(unsigned int size) {
  struct BitfieldHeader h = {};
  h.field16 = (unsigned long)(size << 1U);
}

void explicit_cast_wider_to_bitfield(unsigned int size) {
  struct BitfieldHeader h = {};
  h.field16 = (unsigned short)(size << 1U);
}

void explicit_cast_wider_to_narrower(unsigned int size) {
  struct BitfieldHeader h = {};
  h.field16 = (unsigned char)(size << 1U);
}

// Source (unsigned int, 32-bit) < bit field width (40-bit). Widening — should warn.
void explicit_cast_widen_shift(unsigned int size) {
  struct BitfieldHeader h = {};
  h.field40 = (unsigned long long)(size << 1U);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: either cast from 'unsigned int' to 'unsigned long long'
}

void explicit_cast_widen_multiply(unsigned int size) {
  struct BitfieldHeader h = {};
  h.field40 = (unsigned long long)(size * 2U);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: either cast from 'unsigned int' to 'unsigned long long'
}

// --- Cross-assignment cases ---

// Bit field assigned to a normal (non-bit field) variable.
// h.field8 has declared type 'unsigned int' (32-bit), so h.field8 << 1 is 'unsigned int'.
// Assigning to 'long' (64-bit) is widening.
void bitfield_to_normal_widen(struct BitfieldHeader h) {
  long l;
  l = h.field8 << 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: either cast from 'int' to 'long'
}

void bitfield_to_normal_no_warn(struct BitfieldHeader h) {
  unsigned int i;
  i = h.field8 << 1;
}

// Bit fields of different sizes assigned to each other.
void bitfield_small_to_large(struct BitfieldHeader h) {
  struct BitfieldHeader h2 = {};
  h2.field40 = h.field8 << 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: either cast from 'int' to 'unsigned long long'
}

void bitfield_same_size(struct BitfieldHeader h) {
  struct BitfieldHeader h2 = {};
  h2.field32 = h.field32 << 1;
}

void bitfield_large_to_small(struct BitfieldHeader h) {
  struct BitfieldHeader h2 = {};
  h2.field8 = h.field32 << 1;
}

// --- Signed bit field tests ---

// int is 32 bits on x86, sfield32 is 32-bit signed — no widening.
void sbitfield32_shift(int size) {
  struct BitfieldHeader h = {};
  h.sfield32 = size << 1;
}

void sbitfield32_multiply(int size) {
  struct BitfieldHeader h = {};
  h.sfield32 = size * 2;
}

void sbitfield32_add(int size) {
  struct BitfieldHeader h = {};
  h.sfield32 = size + 1;
}

// short promotes to int (32-bit), sfield16 is 16-bit — truncation, no warning.
void sbitfield16_shift(short size) {
  struct BitfieldHeader h = {};
  h.sfield16 = size << 1;
}

void sbitfield16_multiply(short size) {
  struct BitfieldHeader h = {};
  h.sfield16 = size * 2;
}

void sbitfield16_add(short size) {
  struct BitfieldHeader h = {};
  h.sfield16 = size + 1;
}

// FIXME: Subtraction with short: short promotes to int (32-bit), assigning to long long
// (64-bit) is widening. Checker doesn't warn for '-' (limitation of getMaxCalculationWidth).
void subtract_short_widen(short size) {
  long long l;
  l = size - 1;
}
