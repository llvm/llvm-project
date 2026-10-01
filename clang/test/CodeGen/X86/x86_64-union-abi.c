// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fexperimental-abi-lowering %s -o - | FileCheck %s

// A union member covering the tail makes the whole eightbyte data, so the
// coercion spans the union.
union NarrowStorage {
  int i;
  char c[8];
};

void take_narrow_storage(union NarrowStorage u);
void call_narrow_storage(union NarrowStorage u) { take_narrow_storage(u); }

// CHECK-DAG: declare void @take_narrow_storage(i64)

// Nothing covers the tail of an over-aligned union, so those bytes are padding
// and the coercion is the int alone rather than the full eightbyte.
union OverAligned {
  int i;
} __attribute__((aligned(16)));

void take_over_aligned(union OverAligned u);
void call_over_aligned(union OverAligned u) { take_over_aligned(u); }

// CHECK-DAG: declare void @take_over_aligned(i32)

// Returning one runs the same walk from the return classifier.
union OverAligned ret_over_aligned(void);
void call_ret_over_aligned(void) { ret_over_aligned(); }

// CHECK-DAG: declare i32 @ret_over_aligned()

// Same shape at 8-byte alignment, where the padded tail sits inside the single
// eightbyte rather than beyond it.
union OverAligned8 {
  short s;
} __attribute__((aligned(8)));

void take_over_aligned8(union OverAligned8 u);
void call_over_aligned8(union OverAligned8 u) { take_over_aligned8(u); }

// CHECK-DAG: declare void @take_over_aligned8(i16)

// A 16-byte union whose array member covers both eightbytes is passed in two
// integer registers.
union TwoEightbytes {
  long l;
  char c[16];
};

void take_two_eightbytes(union TwoEightbytes u);
void call_two_eightbytes(union TwoEightbytes u) { take_two_eightbytes(u); }

// CHECK-DAG: declare void @take_two_eightbytes(i64, i64)

// A union sitting at a nonzero offset inside a struct reaches the field walk
// with an already-shifted range, so its padded tail must still read as padding
// on the second eightbyte.
struct WrapsOverAligned {
  double d;
  union OverAligned8 u;
};

void take_wraps(struct WrapsOverAligned s);
void call_wraps(struct WrapsOverAligned s) { take_wraps(s); }

// CHECK-DAG: declare void @take_wraps(double, i16)

// An unnamed bitfield is a member like any other, not skipped.
union WideUnnamedBitfield {
  char c;
  long : 64;
};

void take_wide_unnamed(union WideUnnamedBitfield u);
void call_wide_unnamed(union WideUnnamedBitfield u) { take_wide_unnamed(u); }

// CHECK-DAG: declare void @take_wide_unnamed(i64)

// A non-zero-width unnamed bitfield is INTEGER, which beats the double's SSE
// in the merge, so the union travels in a GPR.
union DoubleUnnamedBitfield {
  double d;
  long : 64;
};

void take_double_unnamed(union DoubleUnnamedBitfield u);
void call_double_unnamed(union DoubleUnnamedBitfield u) {
  take_double_unnamed(u);
}

// CHECK-DAG: declare void @take_double_unnamed(i64)

union DoubleUnnamedBitfield ret_double_unnamed(void);
void call_ret_double_unnamed(void) { ret_double_unnamed(); }

// CHECK-DAG: declare i64 @ret_double_unnamed()
