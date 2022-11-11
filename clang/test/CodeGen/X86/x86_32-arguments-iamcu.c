// RUN: %clang_cc1 -w -triple i386-pc-elfiamcu -mfloat-abi soft -emit-llvm -o - %s | FileCheck %s

// CHECK-LABEL: define{{.*}} void @ints(i32 noundef %a, i32 noundef %b, i32 noundef %c, i32 noundef %d)
void ints(int a, int b, int c, int d) {}

// CHECK-LABEL: define{{.*}} void @floats(float noundef %a, float noundef %b, float noundef %c, float noundef %d)
void floats(float a, float b, float c, float d) {}

// CHECK-LABEL: define{{.*}} void @mixed(i32 noundef %a, float noundef %b, i32 noundef %c, float noundef %d)
void mixed(int a, float b, int c, float d) {}

// CHECK-LABEL: define{{.*}} void @doubles(double noundef %d1, double noundef %d2)
void doubles(double d1, double d2) {}

// CHECK-LABEL: define{{.*}} void @mixedDoubles(i32 noundef %a, double noundef %d1)
void mixedDoubles(int a, double d1) {}

typedef struct st3_t {
  char a[3];
} st3_t;

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

// CHECK-LABEL: define{{.*}} void @smallStructs(i32 %st1.coerce, i32 %st2.coerce, i32 %st3.coerce)
void smallStructs(st4_t st1, st4_t st2, st4_t st3) {}

// CHECK-LABEL: define{{.*}} void @paddedStruct(i32 noundef %i1, i32 %st.coerce0, i32 %st.coerce1, i32 %st4.0)
void paddedStruct(int i1, st5_t st, st4_t st4) {}

// CHECK-LABEL: define{{.*}} void @largeStructBegin(ptr noundef byval(%struct.st12_t) align 4 %st)
void largeStructBegin(st12_t st) {}

// CHECK-LABEL: define{{.*}} void @largeStructMiddle(i32 noundef %i1, ptr noundef byval(%struct.st12_t) align 4 %st, i32 noundef %i2, i32 noundef %i3)
void largeStructMiddle(int i1, st12_t st, int i2, int i3) {}

// CHECK-LABEL: define{{.*}} void @largeStructEnd(i32 noundef %i1, i32 noundef %i2, i32 noundef %i3, i32 %st.0, i32 %st.1, i32 %st.2)
void largeStructEnd(int i1, int i2, int i3, st12_t st) {}

// CHECK-LABEL: define{{.*}} i24 @retNonPow2Struct(i32 %r.coerce)
st3_t retNonPow2Struct(st3_t r) { return r; }

// CHECK-LABEL: define{{.*}} i32 @retSmallStruct(i32 %r.coerce)
st4_t retSmallStruct(st4_t r) { return r; }

// CHECK-LABEL: define{{.*}} i64 @retPaddedStruct(i32 %r.coerce0, i32 %r.coerce1)
st5_t retPaddedStruct(st5_t r) { return r; }

// CHECK-LABEL: define{{.*}} void @retLargeStruct(ptr noalias sret(%struct.st12_t) align 4 %agg.result, i32 noundef %i1, ptr noundef byval(%struct.st12_t) align 4 %r)
st12_t retLargeStruct(int i1, st12_t r) { return r; }

// CHECK-LABEL: define{{.*}} i32 @varArgs(i32 noundef %i1, ...)
int varArgs(int i1, ...) { return i1; }

// CHECK-LABEL: define{{.*}} double @longDoubleArg(double noundef %ld1)
long double longDoubleArg(long double ld1) { return ld1; }

