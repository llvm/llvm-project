// REQUIRES: arm-registered-target
// RUN: %clang_cc1 -no-opaque-pointers -triple armv7---eabi -target-abi aapcs -mfloat-abi hard -emit-llvm %s -o - | FileCheck %s

// RUN: %clang_cc1 -no-opaque-pointers -triple arm64-apple-darwin9 -target-abi darwinpcs \
// RUN:  -ffreestanding -emit-llvm -w -o - %s | FileCheck -check-prefix=CHECK64 %s

// RUN: %clang_cc1 -no-opaque-pointers -triple arm64-linux-gnu -ffreestanding -emit-llvm -w -o - %s \
// RUN:   | FileCheck --check-prefix=CHECK64 %s
typedef long long int64_t;
typedef unsigned int uint32_t;

/* This is not a homogenous aggregate - fundamental types are different */
typedef union {
  float       f[4];
  uint32_t    i[4];
} union_with_first_floats;
union_with_first_floats g_u_f;

extern void takes_union_with_first_floats(union_with_first_floats a);
extern union_with_first_floats returns_union_with_first_floats(void);

void test_union_with_first_floats(void) {
  takes_union_with_first_floats(g_u_f);
}
// CHECK: declare arm_aapcs_vfpcc void @takes_union_with_first_floats([4 x i32])

void test_return_union_with_first_floats(void) {
  g_u_f = returns_union_with_first_floats();
}
// CHECK: declare arm_aapcs_vfpcc void @returns_union_with_first_floats(%union.union_with_first_floats* sret(%union.union_with_first_floats) align 4)

/* This is not a homogenous aggregate - fundamental types are different */
typedef union {
    uint32_t    i[4];
    float       f[4];
} union_with_non_first_floats;
union_with_non_first_floats g_u_nf_f;

extern void takes_union_with_non_first_floats(union_with_non_first_floats a);
extern union_with_non_first_floats returns_union_with_non_first_floats(void);

void test_union_with_non_first_floats(void) {
  takes_union_with_non_first_floats(g_u_nf_f);
}
// CHECK: declare arm_aapcs_vfpcc void @takes_union_with_non_first_floats([4 x i32])

void test_return_union_with_non_first_floats(void) {
  g_u_nf_f = returns_union_with_non_first_floats();
}
// CHECK: declare arm_aapcs_vfpcc void @returns_union_with_non_first_floats(%union.union_with_non_first_floats* sret(%union.union_with_non_first_floats) align 4)

/* This is not a homogenous aggregate - fundamental types are different */
typedef struct {
  float a;
  union_with_first_floats b;
} struct_with_union_with_first_floats;
struct_with_union_with_first_floats g_s_f;

extern void takes_struct_with_union_with_first_floats(struct_with_union_with_first_floats a);
extern struct_with_union_with_first_floats returns_struct_with_union_with_first_floats(void);

void test_struct_with_union_with_first_floats(void) {
  takes_struct_with_union_with_first_floats(g_s_f);
}
// CHECK: declare arm_aapcs_vfpcc void @takes_struct_with_union_with_first_floats([5 x i32])

void test_return_struct_with_union_with_first_floats(void) {
  g_s_f = returns_struct_with_union_with_first_floats();
}
// CHECK: declare arm_aapcs_vfpcc void @returns_struct_with_union_with_first_floats(%struct.struct_with_union_with_first_floats* sret(%struct.struct_with_union_with_first_floats) align 4)

/* This is not a homogenous aggregate - fundamental types are different */
typedef struct {
  float a;
  union_with_non_first_floats b;
} struct_with_union_with_non_first_floats;
struct_with_union_with_non_first_floats g_s_nf_f;

extern void takes_struct_with_union_with_non_first_floats(struct_with_union_with_non_first_floats a);
extern struct_with_union_with_non_first_floats returns_struct_with_union_with_non_first_floats(void);

void test_struct_with_union_with_non_first_floats(void) {
  takes_struct_with_union_with_non_first_floats(g_s_nf_f);
}
// CHECK: declare arm_aapcs_vfpcc void @takes_struct_with_union_with_non_first_floats([5 x i32])

void test_return_struct_with_union_with_non_first_floats(void) {
  g_s_nf_f = returns_struct_with_union_with_non_first_floats();
}
// CHECK: declare arm_aapcs_vfpcc void @returns_struct_with_union_with_non_first_floats(%struct.struct_with_union_with_non_first_floats* sret(%struct.struct_with_union_with_non_first_floats) align 4)

/* Plain array is not a homogenous aggregate */
extern void takes_array_of_floats(float a[4]);
void test_array_of_floats(void) {
  float a[4] = {1.0, 2.0, 3.0, 4.0};
  takes_array_of_floats(a);
}
// CHECK: declare arm_aapcs_vfpcc void @takes_array_of_floats(float* noundef)

/* Struct-type homogenous aggregate */
typedef struct {
  float x, y, z, w;
} struct_with_fundamental_elems;
struct_with_fundamental_elems g_s;

extern void takes_struct_with_fundamental_elems(struct_with_fundamental_elems a);
extern struct_with_fundamental_elems returns_struct_with_fundamental_elems(void);

void test_struct_with_fundamental_elems(void) {
  takes_struct_with_fundamental_elems(g_s);
// CHECK:  call arm_aapcs_vfpcc  void @takes_struct_with_fundamental_elems(%struct.struct_with_fundamental_elems {{.*}})
}
// CHECK: declare arm_aapcs_vfpcc void @takes_struct_with_fundamental_elems(%struct.struct_with_fundamental_elems)

void test_return_struct_with_fundamental_elems(void) {
  g_s = returns_struct_with_fundamental_elems();
// CHECK: call arm_aapcs_vfpcc  %struct.struct_with_fundamental_elems @returns_struct_with_fundamental_elems()
}
// CHECK: declare arm_aapcs_vfpcc %struct.struct_with_fundamental_elems @returns_struct_with_fundamental_elems()

/* Array-type homogenous aggregate */
typedef struct {
  float xyzw[4];
} struct_with_array;
struct_with_array g_s_a;

extern void takes_struct_with_array(struct_with_array a);
extern struct_with_array returns_struct_with_array(void);

void test_struct_with_array(void) {
  takes_struct_with_array(g_s_a);
// CHECK:   call arm_aapcs_vfpcc  void @takes_struct_with_array(%struct.struct_with_array {{.*}})
}
// CHECK: declare arm_aapcs_vfpcc void @takes_struct_with_array(%struct.struct_with_array)

void test_return_struct_with_array(void) {
  g_s_a = returns_struct_with_array();
// CHECK:   call arm_aapcs_vfpcc  %struct.struct_with_array @returns_struct_with_array()
}
// CHECK: declare arm_aapcs_vfpcc %struct.struct_with_array @returns_struct_with_array()

/* This union is a homogenous aggregate. Check that it's passed properly */
typedef union {
  struct_with_fundamental_elems xyzw;
  float a[3];
} union_with_struct_with_fundamental_elems;
union_with_struct_with_fundamental_elems g_u_s_fe;

extern void takes_union_with_struct_with_fundamental_elems(union_with_struct_with_fundamental_elems a);
extern union_with_struct_with_fundamental_elems returns_union_with_struct_with_fundamental_elems(void);

void test_union_with_struct_with_fundamental_elems(void) {
  takes_union_with_struct_with_fundamental_elems(g_u_s_fe);
// CHECK: call arm_aapcs_vfpcc  void @takes_union_with_struct_with_fundamental_elems(%union.union_with_struct_with_fundamental_elems {{.*}})
}
// CHECK: declare arm_aapcs_vfpcc void @takes_union_with_struct_with_fundamental_elems(%union.union_with_struct_with_fundamental_elems)

void test_return_union_with_struct_with_fundamental_elems(void) {
  g_u_s_fe = returns_union_with_struct_with_fundamental_elems();
// CHECK: call arm_aapcs_vfpcc  %union.union_with_struct_with_fundamental_elems @returns_union_with_struct_with_fundamental_elems()
}
// CHECK: declare arm_aapcs_vfpcc %union.union_with_struct_with_fundamental_elems @returns_union_with_struct_with_fundamental_elems()

// Make sure HAs that can be partially fit into VFP registers will be allocated
// on stack and that later VFP candidates will go on stack as well.
typedef struct {
  double x;
  double a2;
  double a3;
  double a4;
} struct_of_four_doubles;
extern void takes_struct_of_four_doubles(double a, struct_of_four_doubles b, struct_of_four_doubles c, double d);
struct_of_four_doubles g_s4d;

void test_struct_of_four_doubles(void) {
// CHECK: test_struct_of_four_doubles
// CHECK: call arm_aapcs_vfpcc void @takes_struct_of_four_doubles(double {{.*}}, %struct.struct_of_four_doubles {{.*}}, %struct.struct_of_four_doubles {{.*}}, double {{.*}})
// CHECK64: test_struct_of_four_doubles
// CHECK64: call void @takes_struct_of_four_doubles(double {{.*}}, [4 x double] {{.*}}, [4 x double] {{.*}}, double {{.*}})
  takes_struct_of_four_doubles(3.0, g_s4d, g_s4d, 4.0);
}

extern void takes_struct_of_four_doubles_variadic(double a, struct_of_four_doubles b, struct_of_four_doubles c, double d, ...);

void test_struct_of_four_doubles_variadic(void) {
// CHECK: test_struct_of_four_doubles_variadic
// CHECK: call arm_aapcs_vfpcc void (double, [4 x i64], [4 x i64], double, ...) @takes_struct_of_four_doubles_variadic(double {{.*}}, [4 x i64] {{.*}}, [4 x i64] {{.*}}, double {{.*}})
  takes_struct_of_four_doubles_variadic(3.0, g_s4d, g_s4d, 4.0);
}

extern void takes_struct_with_backfill(float f1, double a, float f2, struct_of_four_doubles b, struct_of_four_doubles c, double d);
void test_struct_with_backfill(void) {
// CHECK: test_struct_with_backfill
// CHECK: call arm_aapcs_vfpcc void @takes_struct_with_backfill(float {{.*}}, double {{.*}}, float {{.*}}, %struct.struct_of_four_doubles {{.*}}, %struct.struct_of_four_doubles {{.*}}, double {{.*}})
  takes_struct_with_backfill(3.0, 3.1, 3.2, g_s4d, g_s4d, 4.0);
}

typedef __attribute__(( ext_vector_type(8) )) char __char8;
typedef __attribute__(( ext_vector_type(4) ))  short __short4;
typedef struct {
  __char8  a1;
  __short4 a2;
  __char8  a3;
  __short4 a4;
} struct_of_vecs;
extern void takes_struct_of_vecs(double a, struct_of_vecs b, struct_of_vecs c, double d);
struct_of_vecs g_vec;

void test_struct_of_vecs(void) {
// CHECK: test_struct_of_vecs
// CHECK: call arm_aapcs_vfpcc void @takes_struct_of_vecs(double {{.*}}, %struct.struct_of_vecs {{.*}}, %struct.struct_of_vecs {{.*}}, double {{.*}})
// CHECK64: test_struct_of_vecs
// CHECK64: call void @takes_struct_of_vecs(double {{.*}}, [4 x <8 x i8>] {{.*}}, [4 x <8 x i8>] {{.*}}, double {{.*}})
  takes_struct_of_vecs(3.0, g_vec, g_vec, 4.0);
}

typedef struct {
  double a;
  long double b;
} struct_of_double_and_long_double;
struct_of_double_and_long_double g_dld;

struct_of_double_and_long_double test_struct_of_double_and_long_double(void) {
  return g_dld;
}
// CHECK: define{{.*}} arm_aapcs_vfpcc %struct.struct_of_double_and_long_double @test_struct_of_double_and_long_double()

// FIXME: Tests necessary:
//         - Vectors
//         - C++ stuff
