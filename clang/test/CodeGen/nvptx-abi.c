// RUN: %clang_cc1 -triple nvptx-unknown-unknown -o - %s -emit-llvm | FileCheck %s
// RUN: %clang_cc1 -triple nvptx64-unknown-unknown -o - %s -emit-llvm | FileCheck %s --check-prefixes=CHECK,NVPTX64

typedef struct float4_s {
  float x, y, z, w;
} float4_t;

float4_t my_function(void) {
// CHECK-LABEL: define{{.*}} %struct.float4_s @my_function
  float4_t t;
  return t;
};

float bar(void) {
  float4_t ret;
// CHECK-LABEL: @bar
// CHECK: call %struct.float4_s @my_function
  ret = my_function();
  return ret.x;
}

void foo(float4_t x) {
// CHECK-LABEL: @foo
// CHECK: ptr noundef byval(%struct.float4_s) align 4 %x
}

void fooN(float4_t x, float4_t y, float4_t z) {
// CHECK-LABEL: @fooN
// CHECK: ptr noundef byval(%struct.float4_s) align 4 %x
// CHECK: ptr noundef byval(%struct.float4_s) align 4 %y
// CHECK: ptr noundef byval(%struct.float4_s) align 4 %z
}

typedef struct nested_s {
  unsigned long long x;
  float z[64];
  float4_t t;
} nested_t;

void baz(nested_t x) {
// CHECK-LABEL: @baz
// CHECK: ptr noundef byval(%struct.nested_s) align 8 %x)
}

// A union is lowered to the LLVM type of one member, here pad_t, whose padding
// holds data of the other member. Aggregates containing a union are returned
// as an integer array so that those bytes are not dropped.
typedef struct pad_s {
  short s;
  unsigned long long l;
} pad_t;

typedef union pad_u {
  unsigned long long l;
  pad_t p;
} pad_u_t;

pad_u_t ret_union(void) {
// CHECK-LABEL: define{{.*}} [2 x i64] @ret_union
// CHECK: load [2 x i64], ptr
// CHECK: ret [2 x i64]
  pad_u_t u = {0x1122334455667788ULL};
  return u;
}

unsigned long long call_ret_union(void) {
// CHECK-LABEL: @call_ret_union
// CHECK: call [2 x i64] @ret_union
// CHECK: store [2 x i64]
  return ret_union().l;
}

typedef struct with_union_s {
  int i;
  pad_u_t u;
} with_union_t;

with_union_t ret_with_union(void) {
// CHECK-LABEL: define{{.*}} [3 x i64] @ret_with_union
  with_union_t t;
  return t;
}

typedef struct with_union_array_s {
  pad_u_t u[2];
} with_union_array_t;

with_union_array_t ret_with_union_array(void) {
// CHECK-LABEL: define{{.*}} [4 x i64] @ret_with_union_array
  with_union_array_t t;
  return t;
}

// Keep the ABI alignment of the original LLVM return type, which can differ
// from the source type's explicit alignment.
typedef struct __attribute__((aligned(16))) {
  int x;
} aligned16_t;
typedef union {
  aligned16_t s;
  unsigned char bytes[16];
} union_align16_t;

union_align16_t ret_align16(void) {
// CHECK-LABEL: define{{.*}} [4 x i32] @ret_align16
  union_align16_t u = {0};
  return u;
}

typedef struct __attribute__((aligned(8))) {
  short x;
} aligned8_t;
typedef union {
  aligned8_t s;
  unsigned char bytes[8];
} union_align8_t;

union_align8_t ret_align8(void) {
// CHECK-LABEL: define{{.*}} [4 x i16] @ret_align8
  union_align8_t u = {0};
  return u;
}

#if __SIZEOF_POINTER__ == 8
typedef union {
  __int128 x;
  unsigned char bytes[16];
} union_i128_t;

union_i128_t ret_i128(void) {
// NVPTX64-LABEL: define{{.*}} [1 x i128] @ret_i128
  union_i128_t u = {0};
  return u;
}
#endif
