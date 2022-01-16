// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -S -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple r600-unknown-unknown -S -emit-llvm -o - %s | FileCheck %s

typedef __attribute__(( ext_vector_type(2) )) char char2;
typedef __attribute__(( ext_vector_type(3) )) char char3;
typedef __attribute__(( ext_vector_type(4) )) char char4;

typedef __attribute__(( ext_vector_type(2) )) short short2;
typedef __attribute__(( ext_vector_type(3) )) short short3;
typedef __attribute__(( ext_vector_type(4) )) short short4;

typedef __attribute__(( ext_vector_type(2) )) int int2;
typedef __attribute__(( ext_vector_type(3) )) int int3;
typedef __attribute__(( ext_vector_type(4) )) int int4;
typedef __attribute__(( ext_vector_type(16) )) int int16;
typedef __attribute__(( ext_vector_type(32) )) int int32;

// CHECK: %struct.empty_struct = type {}
typedef struct empty_struct
{
} empty_struct;

// CHECK-NOT: %struct.single_element_struct_arg
typedef struct single_element_struct_arg
{
    int i;
} single_element_struct_arg_t;

// CHECK-NOT: %struct.nested_single_element_struct_arg
typedef struct nested_single_element_struct_arg
{
  single_element_struct_arg_t i;
} nested_single_element_struct_arg_t;

// CHECK: %struct.struct_arg = type { i32, float, i32 }
typedef struct struct_arg
{
    int i1;
    float f;
    int i2;
} struct_arg_t;

// CHECK: %struct.struct_padding_arg = type { i8, i64 }
typedef struct struct_padding_arg
{
  char i1;
  long f;
} struct_padding_arg;

// CHECK: %struct.struct_of_arrays_arg = type { [2 x i32], float, [4 x i32], [3 x float], i32 }
typedef struct struct_of_arrays_arg
{
    int i1[2];
    float f1;
    int i2[4];
    float f2[3];
    int i3;
} struct_of_arrays_arg_t;

// CHECK: %struct.struct_of_structs_arg = type { i32, float, %struct.struct_arg, i32 }
typedef struct struct_of_structs_arg
{
    int i1;
    float f1;
    struct_arg_t s1;
    int i2;
} struct_of_structs_arg_t;

typedef union
{
  int b1;
  float b2;
} transparent_u __attribute__((__transparent_union__));

// CHECK: %struct.single_array_element_struct_arg = type { [4 x i32] }
typedef struct single_array_element_struct_arg
{
    int i[4];
} single_array_element_struct_arg_t;

// CHECK: %struct.single_struct_element_struct_arg = type { %struct.inner }
// CHECK: %struct.inner = type { i32, i64 }
typedef struct single_struct_element_struct_arg
{
  struct inner {
    int a;
    long b;
  } s;
} single_struct_element_struct_arg_t;

// CHECK: %struct.different_size_type_pair
typedef struct different_size_type_pair {
  long l;
  int i;
} different_size_type_pair;

// CHECK: %struct.flexible_array = type { i32, [0 x i32] }
typedef struct flexible_array
{
  int i;
  int flexible[];
} flexible_array;

// CHECK: %struct.struct_arr16 = type { [16 x i32] }
typedef struct struct_arr16
{
    int arr[16];
} struct_arr16;

// CHECK: %struct.struct_arr32 = type { [32 x i32] }
typedef struct struct_arr32
{
    int arr[32];
} struct_arr32;

// CHECK: %struct.struct_arr33 = type { [33 x i32] }
typedef struct struct_arr33
{
    int arr[33];
} struct_arr33;

// CHECK: %struct.struct_char_arr32 = type { [32 x i8] }
typedef struct struct_char_arr32
{
  char arr[32];
} struct_char_arr32;

// CHECK-NOT: %struct.struct_char_x8
typedef struct struct_char_x8 {
  char x, y, z, w;
  char a, b, c, d;
} struct_char_x8;

// CHECK-NOT: %struct.struct_char_x4
typedef struct struct_char_x4 {
  char x, y, z, w;
} struct_char_x4;

// CHECK-NOT: %struct.struct_char_x3
typedef struct struct_char_x3 {
  char x, y, z;
} struct_char_x3;

// CHECK-NOT: %struct.struct_char_x2
typedef struct struct_char_x2 {
  char x, y;
} struct_char_x2;

// CHECK-NOT: %struct.struct_char_x1
typedef struct struct_char_x1 {
  char x;
} struct_char_x1;

// 4 registers from fields, 5 if padding included.
// CHECK: %struct.nested = type { i8, i64 }
// CHECK: %struct.num_regs_nested_struct = type { i32, %struct.nested }
typedef struct num_regs_nested_struct {
  int x;
  struct nested {
    char z;
    long y;
  } inner;
} num_regs_nested_struct;

// CHECK: %struct.double_nested = type { %struct.inner_inner }
// CHECK: %struct.inner_inner = type { i8, i32, i8 }
// CHECK: %struct.double_nested_struct = type { i32, %struct.double_nested, i16 }
typedef struct double_nested_struct {
  int x;
  struct double_nested {
    struct inner_inner {
      char y;
      int q;
      char z;
    } inner_inner;
  } inner;

  short w;
} double_nested_struct;

// This is a large struct, but uses fewer registers than the limit.
// CHECK: %struct.large_struct_padding = type { i8, i32, i8, i32, i8, i8, i16, i16, [3 x i8], i64, i32, i8, i32, i16, i8 }
typedef struct large_struct_padding {
  char e0;
  int e1;
  char e2;
  int e3;
  char e4;
  char e5;
  short e6;
  short e7;
  char e8[3];
  long e9;
  int e10;
  char e11;
  int e12;
  short e13;
  char e14;
} large_struct_padding;

// CHECK: %struct.int3_pair = type { <3 x i32>, <3 x i32> }
// The number of registers computed should be 6, not 8.
typedef struct int3_pair {
	int3 dx;
	int3 dy;
} int3_pair;

// CHECK: %struct.struct_4regs = type { i32, i32, i32, i32 }
typedef struct struct_4regs
{
  int x;
  int y;
  int z;
  int w;
} struct_4regs;

// CHECK: void @kernel_empty_struct_arg(%struct.empty_struct %s.coerce)
__kernel void kernel_empty_struct_arg(empty_struct s) { }

// CHECK: void @kernel_single_element_struct_arg(i32 %arg1.coerce)
__kernel void kernel_single_element_struct_arg(single_element_struct_arg_t arg1) { }

// CHECK: void @kernel_nested_single_element_struct_arg(i32 %arg1.coerce)
__kernel void kernel_nested_single_element_struct_arg(nested_single_element_struct_arg_t arg1) { }

// CHECK: void @kernel_struct_arg(%struct.struct_arg %arg1.coerce)
__kernel void kernel_struct_arg(struct_arg_t arg1) { }

// CHECK: void @kernel_struct_padding_arg(%struct.struct_padding_arg %arg1.coerce)
__kernel void kernel_struct_padding_arg(struct_padding_arg arg1) { }

// CHECK: void @kernel_test_struct_of_arrays_arg(%struct.struct_of_arrays_arg %arg1.coerce)
__kernel void kernel_test_struct_of_arrays_arg(struct_of_arrays_arg_t arg1) { }

// CHECK: void @kernel_struct_of_structs_arg(%struct.struct_of_structs_arg %arg1.coerce)
__kernel void kernel_struct_of_structs_arg(struct_of_structs_arg_t arg1) { }

// CHECK: void @test_kernel_transparent_union_arg(i32 %u.coerce)
__kernel void test_kernel_transparent_union_arg(transparent_u u) { }

// CHECK: void @kernel_single_array_element_struct_arg(%struct.single_array_element_struct_arg %arg1.coerce)
__kernel void kernel_single_array_element_struct_arg(single_array_element_struct_arg_t arg1) { }

// CHECK: void @kernel_single_struct_element_struct_arg(%struct.single_struct_element_struct_arg %arg1.coerce)
__kernel void kernel_single_struct_element_struct_arg(single_struct_element_struct_arg_t arg1) { }

// CHECK: void @kernel_different_size_type_pair_arg(%struct.different_size_type_pair %arg1.coerce)
__kernel void kernel_different_size_type_pair_arg(different_size_type_pair arg1) { }

// CHECK: define{{.*}} void @func_f32_arg(float noundef %arg)
void func_f32_arg(float arg) { }

// CHECK: define{{.*}} void @func_v2i16_arg(<2 x i16> noundef %arg)
void func_v2i16_arg(short2 arg) { }

// CHECK: define{{.*}} void @func_v3i32_arg(<3 x i32> noundef %arg)
void func_v3i32_arg(int3 arg) { }

// CHECK: define{{.*}} void @func_v4i32_arg(<4 x i32> noundef %arg)
void func_v4i32_arg(int4 arg) { }

// CHECK: define{{.*}} void @func_v16i32_arg(<16 x i32> noundef %arg)
void func_v16i32_arg(int16 arg) { }

// CHECK: define{{.*}} void @func_v32i32_arg(<32 x i32> noundef %arg)
void func_v32i32_arg(int32 arg) { }

// CHECK: define{{.*}} void @func_empty_struct_arg()
void func_empty_struct_arg(empty_struct empty) { }

// CHECK: void @func_single_element_struct_arg(i32 %arg1.coerce)
void func_single_element_struct_arg(single_element_struct_arg_t arg1) { }

// CHECK: void @func_nested_single_element_struct_arg(i32 %arg1.coerce)
void func_nested_single_element_struct_arg(nested_single_element_struct_arg_t arg1) { }

// CHECK: void @func_struct_arg(i32 %arg1.coerce0, float %arg1.coerce1, i32 %arg1.coerce2)
void func_struct_arg(struct_arg_t arg1) { }

// CHECK: void @func_struct_padding_arg(i8 %arg1.coerce0, i64 %arg1.coerce1)
void func_struct_padding_arg(struct_padding_arg arg1) { }

// CHECK: define{{.*}} void @func_struct_char_x8([2 x i32] %arg.coerce)
void func_struct_char_x8(struct_char_x8 arg) { }

// CHECK: define{{.*}} void @func_struct_char_x4(i32 %arg.coerce)
void func_struct_char_x4(struct_char_x4 arg) { }

// CHECK: define{{.*}} void @func_struct_char_x3(i32 %arg.coerce)
void func_struct_char_x3(struct_char_x3 arg) { }

// CHECK: define{{.*}} void @func_struct_char_x2(i16 %arg.coerce)
void func_struct_char_x2(struct_char_x2 arg) { }

// CHECK: define{{.*}} void @func_struct_char_x1(i8 %arg.coerce)
void func_struct_char_x1(struct_char_x1 arg) { }

// CHECK: void @func_transparent_union_arg(i32 %u.coerce)
void func_transparent_union_arg(transparent_u u) { }

// CHECK: void @func_single_array_element_struct_arg([4 x i32] %arg1.coerce)
void func_single_array_element_struct_arg(single_array_element_struct_arg_t arg1) { }

// CHECK: void @func_single_struct_element_struct_arg(%struct.inner %arg1.coerce)
void func_single_struct_element_struct_arg(single_struct_element_struct_arg_t arg1) { }

// CHECK: void @func_different_size_type_pair_arg(i64 %arg1.coerce0, i32 %arg1.coerce1)
void func_different_size_type_pair_arg(different_size_type_pair arg1) { }

// CHECK: void @func_flexible_array_arg(%struct.flexible_array addrspace(5)* nocapture noundef byval(%struct.flexible_array) align 4 %arg)
void func_flexible_array_arg(flexible_array arg) { }

// CHECK: define{{.*}} float @func_f32_ret()
float func_f32_ret()
{
  return 0.0f;
}

// CHECK: define{{.*}} void @func_empty_struct_ret()
empty_struct func_empty_struct_ret()
{
  empty_struct s = {};
  return s;
}

// CHECK: define{{.*}} i32 @single_element_struct_ret()
// CHECK: ret i32 0
single_element_struct_arg_t single_element_struct_ret()
{
  single_element_struct_arg_t s = { 0 };
  return s;
}

// CHECK: define{{.*}} i32 @nested_single_element_struct_ret()
// CHECK: ret i32 0
nested_single_element_struct_arg_t nested_single_element_struct_ret()
{
  nested_single_element_struct_arg_t s = { 0 };
  return s;
}

// CHECK: define{{.*}} %struct.struct_arg @func_struct_ret()
// CHECK: ret %struct.struct_arg zeroinitializer
struct_arg_t func_struct_ret()
{
  struct_arg_t s = { 0 };
  return s;
}

// CHECK: define{{.*}} %struct.struct_padding_arg @func_struct_padding_ret()
// CHECK: ret %struct.struct_padding_arg zeroinitializer
struct_padding_arg func_struct_padding_ret()
{
  struct_padding_arg s = { 0 };
  return s;
}

// CHECK: define{{.*}} [2 x i32] @func_struct_char_x8_ret()
// CHECK: ret [2 x i32] zeroinitializer
struct_char_x8 func_struct_char_x8_ret()
{
  struct_char_x8 s = { 0 };
  return s;
}

// CHECK: define{{.*}} i32 @func_struct_char_x4_ret()
// CHECK: ret i32 0
struct_char_x4 func_struct_char_x4_ret()
{
  struct_char_x4 s = { 0 };
  return s;
}

// CHECK: define{{.*}} i32 @func_struct_char_x3_ret()
// CHECK: ret i32 0
struct_char_x3 func_struct_char_x3_ret()
{
  struct_char_x3 s = { 0 };
  return s;
}

// CHECK: define{{.*}} i16 @func_struct_char_x2_ret()
struct_char_x2 func_struct_char_x2_ret()
{
  struct_char_x2 s = { 0 };
  return s;
}

// CHECK: define{{.*}} i8 @func_struct_char_x1_ret()
// CHECK: ret i8 0
struct_char_x1 func_struct_char_x1_ret()
{
  struct_char_x1 s = { 0 };
  return s;
}

// CHECK: define{{.*}} %struct.struct_arr16 @func_ret_struct_arr16()
// CHECK: ret %struct.struct_arr16 zeroinitializer
struct_arr16 func_ret_struct_arr16()
{
  struct_arr16 s = { 0 };
  return s;
}

// CHECK: define{{.*}} void @func_ret_struct_arr32(%struct.struct_arr32 addrspace(5)* noalias nocapture writeonly sret(%struct.struct_arr32) align 4 %agg.result)
struct_arr32 func_ret_struct_arr32()
{
  struct_arr32 s = { 0 };
  return s;
}

// CHECK: define{{.*}} void @func_ret_struct_arr33(%struct.struct_arr33 addrspace(5)* noalias nocapture writeonly sret(%struct.struct_arr33) align 4 %agg.result)
struct_arr33 func_ret_struct_arr33()
{
  struct_arr33 s = { 0 };
  return s;
}

// CHECK: define{{.*}} %struct.struct_char_arr32 @func_ret_struct_char_arr32()
struct_char_arr32 func_ret_struct_char_arr32()
{
  struct_char_arr32 s = { 0 };
  return s;
}

// CHECK: define{{.*}} i32 @func_transparent_union_ret() local_unnamed_addr #1 {
// CHECK: ret i32 0
transparent_u func_transparent_union_ret()
{
  transparent_u u = { 0 };
  return u;
}

// CHECK: define{{.*}} %struct.different_size_type_pair @func_different_size_type_pair_ret()
different_size_type_pair func_different_size_type_pair_ret()
{
  different_size_type_pair s = { 0 };
  return s;
}

// CHECK: define{{.*}} void @func_flexible_array_ret(%struct.flexible_array addrspace(5)* noalias nocapture writeonly sret(%struct.flexible_array) align 4 %agg.result)
flexible_array func_flexible_array_ret()
{
  flexible_array s = { 0 };
  return s;
}

// CHECK: define{{.*}} void @func_reg_state_lo(<4 x i32> noundef %arg0, <4 x i32> noundef %arg1, <4 x i32> noundef %arg2, i32 noundef %arg3, i32 %s.coerce0, float %s.coerce1, i32 %s.coerce2)
void func_reg_state_lo(int4 arg0, int4 arg1, int4 arg2, int arg3, struct_arg_t s) { }

// CHECK: define{{.*}} void @func_reg_state_hi(<4 x i32> noundef %arg0, <4 x i32> noundef %arg1, <4 x i32> noundef %arg2, i32 noundef %arg3, i32 noundef %arg4, %struct.struct_arg addrspace(5)* nocapture noundef byval(%struct.struct_arg) align 4 %s)
void func_reg_state_hi(int4 arg0, int4 arg1, int4 arg2, int arg3, int arg4, struct_arg_t s) { }

// XXX - Why don't the inner structs flatten?
// CHECK: define{{.*}} void @func_reg_state_num_regs_nested_struct(<4 x i32> noundef %arg0, i32 noundef %arg1, i32 %arg2.coerce0, %struct.nested %arg2.coerce1, i32 %arg3.coerce0, %struct.nested %arg3.coerce1, %struct.num_regs_nested_struct addrspace(5)* nocapture noundef byval(%struct.num_regs_nested_struct) align 8 %arg4)
void func_reg_state_num_regs_nested_struct(int4 arg0, int arg1, num_regs_nested_struct arg2, num_regs_nested_struct arg3, num_regs_nested_struct arg4) { }

// CHECK: define{{.*}} void @func_double_nested_struct_arg(<4 x i32> noundef %arg0, i32 noundef %arg1, i32 %arg2.coerce0, %struct.double_nested %arg2.coerce1, i16 %arg2.coerce2)
void func_double_nested_struct_arg(int4 arg0, int arg1, double_nested_struct arg2) { }

// CHECK: define{{.*}} %struct.double_nested_struct @func_double_nested_struct_ret(<4 x i32> noundef %arg0, i32 noundef %arg1)
double_nested_struct func_double_nested_struct_ret(int4 arg0, int arg1) {
  double_nested_struct s = { 0 };
  return s;
}

// CHECK: define{{.*}} void @func_large_struct_padding_arg_direct(i8 %arg.coerce0, i32 %arg.coerce1, i8 %arg.coerce2, i32 %arg.coerce3, i8 %arg.coerce4, i8 %arg.coerce5, i16 %arg.coerce6, i16 %arg.coerce7, [3 x i8] %arg.coerce8, i64 %arg.coerce9, i32 %arg.coerce10, i8 %arg.coerce11, i32 %arg.coerce12, i16 %arg.coerce13, i8 %arg.coerce14)
void func_large_struct_padding_arg_direct(large_struct_padding arg) { }

// CHECK: define{{.*}} void @func_large_struct_padding_arg_store(%struct.large_struct_padding addrspace(1)* nocapture noundef writeonly %out, %struct.large_struct_padding addrspace(5)* nocapture noundef readonly byval(%struct.large_struct_padding) align 8 %arg)
void func_large_struct_padding_arg_store(global large_struct_padding* out, large_struct_padding arg) {
  *out = arg;
}

// CHECK: define{{.*}} void @v3i32_reg_count(<3 x i32> noundef %arg1, <3 x i32> noundef %arg2, <3 x i32> noundef %arg3, <3 x i32> noundef %arg4, i32 %arg5.coerce0, float %arg5.coerce1, i32 %arg5.coerce2)
void v3i32_reg_count(int3 arg1, int3 arg2, int3 arg3, int3 arg4, struct_arg_t arg5) { }

// Function signature from blender, nothing should be passed byval. The v3i32
// should not count as 4 passed registers.
// CHECK: define{{.*}} void @v3i32_pair_reg_count(%struct.int3_pair addrspace(5)* nocapture noundef %arg0, <3 x i32> %arg1.coerce0, <3 x i32> %arg1.coerce1, <3 x i32> noundef %arg2, <3 x i32> %arg3.coerce0, <3 x i32> %arg3.coerce1, <3 x i32> noundef %arg4, float noundef %arg5)
void v3i32_pair_reg_count(int3_pair *arg0, int3_pair arg1, int3 arg2, int3_pair arg3, int3 arg4, float arg5) { }

// Each short4 should fit pack into 2 registers.
// CHECK: define{{.*}} void @v4i16_reg_count(<4 x i16> noundef %arg0, <4 x i16> noundef %arg1, <4 x i16> noundef %arg2, <4 x i16> noundef %arg3, <4 x i16> noundef %arg4, <4 x i16> noundef %arg5, i32 %arg6.coerce0, i32 %arg6.coerce1, i32 %arg6.coerce2, i32 %arg6.coerce3)
void v4i16_reg_count(short4 arg0, short4 arg1, short4 arg2, short4 arg3,
                     short4 arg4, short4 arg5, struct_4regs arg6) { }

// CHECK: define{{.*}} void @v4i16_pair_reg_count_over(<4 x i16> noundef %arg0, <4 x i16> noundef %arg1, <4 x i16> noundef %arg2, <4 x i16> noundef %arg3, <4 x i16> noundef %arg4, <4 x i16> noundef %arg5, <4 x i16> noundef %arg6, %struct.struct_4regs addrspace(5)* nocapture noundef byval(%struct.struct_4regs) align 4 %arg7)
void v4i16_pair_reg_count_over(short4 arg0, short4 arg1, short4 arg2, short4 arg3,
                               short4 arg4, short4 arg5, short4 arg6, struct_4regs arg7) { }

// CHECK: define{{.*}} void @v3i16_reg_count(<3 x i16> noundef %arg0, <3 x i16> noundef %arg1, <3 x i16> noundef %arg2, <3 x i16> noundef %arg3, <3 x i16> noundef %arg4, <3 x i16> noundef %arg5, i32 %arg6.coerce0, i32 %arg6.coerce1, i32 %arg6.coerce2, i32 %arg6.coerce3)
void v3i16_reg_count(short3 arg0, short3 arg1, short3 arg2, short3 arg3,
                     short3 arg4, short3 arg5, struct_4regs arg6) { }

// CHECK: define{{.*}} void @v3i16_reg_count_over(<3 x i16> noundef %arg0, <3 x i16> noundef %arg1, <3 x i16> noundef %arg2, <3 x i16> noundef %arg3, <3 x i16> noundef %arg4, <3 x i16> noundef %arg5, <3 x i16> noundef %arg6, %struct.struct_4regs addrspace(5)* nocapture noundef byval(%struct.struct_4regs) align 4 %arg7)
void v3i16_reg_count_over(short3 arg0, short3 arg1, short3 arg2, short3 arg3,
                          short3 arg4, short3 arg5, short3 arg6, struct_4regs arg7) { }

// CHECK: define{{.*}} void @v2i16_reg_count(<2 x i16> noundef %arg0, <2 x i16> noundef %arg1, <2 x i16> noundef %arg2, <2 x i16> noundef %arg3, <2 x i16> noundef %arg4, <2 x i16> noundef %arg5, <2 x i16> noundef %arg6, <2 x i16> noundef %arg7, <2 x i16> noundef %arg8, <2 x i16> noundef %arg9, <2 x i16> noundef %arg10, <2 x i16> noundef %arg11, i32 %arg13.coerce0, i32 %arg13.coerce1, i32 %arg13.coerce2, i32 %arg13.coerce3)
void v2i16_reg_count(short2 arg0, short2 arg1, short2 arg2, short2 arg3,
                     short2 arg4, short2 arg5, short2 arg6, short2 arg7,
                     short2 arg8, short2 arg9, short2 arg10, short2 arg11,
                     struct_4regs arg13) { }

// CHECK: define{{.*}} void @v2i16_reg_count_over(<2 x i16> noundef %arg0, <2 x i16> noundef %arg1, <2 x i16> noundef %arg2, <2 x i16> noundef %arg3, <2 x i16> noundef %arg4, <2 x i16> noundef %arg5, <2 x i16> noundef %arg6, <2 x i16> noundef %arg7, <2 x i16> noundef %arg8, <2 x i16> noundef %arg9, <2 x i16> noundef %arg10, <2 x i16> noundef %arg11, <2 x i16> noundef %arg12, %struct.struct_4regs addrspace(5)* nocapture noundef byval(%struct.struct_4regs) align 4 %arg13)
void v2i16_reg_count_over(short2 arg0, short2 arg1, short2 arg2, short2 arg3,
                          short2 arg4, short2 arg5, short2 arg6, short2 arg7,
                          short2 arg8, short2 arg9, short2 arg10, short2 arg11,
                          short2 arg12, struct_4regs arg13) { }

// CHECK: define{{.*}} void @v2i8_reg_count(<2 x i8> noundef %arg0, <2 x i8> noundef %arg1, <2 x i8> noundef %arg2, <2 x i8> noundef %arg3, <2 x i8> noundef %arg4, <2 x i8> noundef %arg5, i32 %arg6.coerce0, i32 %arg6.coerce1, i32 %arg6.coerce2, i32 %arg6.coerce3)
void v2i8_reg_count(char2 arg0, char2 arg1, char2 arg2, char2 arg3,
                    char2 arg4, char2 arg5, struct_4regs arg6) { }

// CHECK: define{{.*}} void @v2i8_reg_count_over(<2 x i8> noundef %arg0, <2 x i8> noundef %arg1, <2 x i8> noundef %arg2, <2 x i8> noundef %arg3, <2 x i8> noundef %arg4, <2 x i8> noundef %arg5, i32 noundef %arg6, %struct.struct_4regs addrspace(5)* nocapture noundef byval(%struct.struct_4regs) align 4 %arg7)
void v2i8_reg_count_over(char2 arg0, char2 arg1, char2 arg2, char2 arg3,
                         char2 arg4, char2 arg5, int arg6, struct_4regs arg7) { }

// CHECK: define{{.*}} void @num_regs_left_64bit_aggregate(<4 x i32> noundef %arg0, <4 x i32> noundef %arg1, <4 x i32> noundef %arg2, <3 x i32> noundef %arg3, [2 x i32] %arg4.coerce, i32 noundef %arg5)
void num_regs_left_64bit_aggregate(int4 arg0, int4 arg1, int4 arg2, int3 arg3, struct_char_x8 arg4, int arg5) { }
