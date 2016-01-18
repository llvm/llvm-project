// RUN: %clang_cc1 -w -triple i386-pc-elfiamcu -mfloat-abi soft -emit-llvm -o - %s | FileCheck %s

// CHECK-LABEL: define void @ints(i32 inreg %a, i32 inreg %b, i32 inreg %c, i32 %d)
void ints(int a, int b, int c, int d) {}

// CHECK-LABEL: define void @floats(float inreg %a, float inreg %b, float inreg %c, float %d)
void floats(float a, float b, float c, float d) {}

// CHECK-LABEL: define void @mixed(i32 inreg %a, float inreg %b, i32 inreg %c, float %d)
void mixed(int a, float b, int c, float d) {}

// CHECK-LABEL: define void @doubles(double inreg %d1, double %d2)
void doubles(double d1, double d2) {}

// CHECK-LABEL: define void @mixedDoubles(i32 inreg %a, double inreg %d1)
void mixedDoubles(int a, double d1) {}

typedef struct st4_t {
  int a;
} st4_t;

typedef struct st5_t {
  int a;
  char b;
} st5_t;

typedef  struct st12_t {
  int a;
  int b;
  int c;
} st12_t;

// CHECK-LABEL: define void @smallStructs(i32 inreg %st1.coerce, i32 inreg %st2.coerce, i32 inreg %st3.coerce)
void smallStructs(st4_t st1, st4_t st2, st4_t st3) {}

// CHECK-LABEL: define void @paddedStruct(i32 inreg %i1, i32 inreg %st.coerce0, i32 inreg %st.coerce1, i32 %st4.0)
void paddedStruct(int i1, st5_t st, st4_t st4) {}

// CHECK-LABEL: define void @largeStruct(i32 %st.0, i32 %st.1, i32 %st.2)
void largeStruct(st12_t st) {}

// CHECK-LABEL: define void @largeStructMiddle(i32 inreg %i1, i32 %st.0, i32 %st.1, i32 %st.2, i32 inreg %i2, i32 inreg %i3)
void largeStructMiddle(int i1, st12_t st, int i2, int i3) {}

// CHECK-LABEL: define i32 @retSmallStruct(i32 inreg %r.coerce)
st4_t retSmallStruct(st4_t r) { return r; }

// CHECK-LABEL: define i64 @retPaddedStruct(i32 inreg %r.coerce0, i32 inreg %r.coerce1)
st5_t retPaddedStruct(st5_t r) { return r; }

// CHECK-LABEL: define void @retLargeStruct(%struct.st12_t* inreg noalias sret %agg.result, i32 inreg %i1, i32 %r.0, i32 %r.1, i32 %r.2)
st12_t retLargeStruct(int i1, st12_t r) { return r; }

// FIXME: We really shouldn't be marking this inreg. Right now the
// inreg gets ignored by the CG for varargs functions, but that's
// insane.
// CHECK-LABEL: define i32 @varArgs(i32 inreg %i1, ...)
int varArgs(int i1, ...) { return i1; }

// CHECK-LABEL: define double @longDoubleArg(double inreg %ld1)
long double longDoubleArg(long double ld1) { return ld1; }

