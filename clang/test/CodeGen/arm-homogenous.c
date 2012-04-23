// RUN: %clang_cc1 -triple armv7---eabi -target-abi aapcs -mfloat-abi hard -emit-llvm %s -o - | FileCheck %s

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
// CHECK: declare arm_aapcs_vfpcc void @returns_union_with_first_floats(%union.union_with_first_floats* sret)

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
// CHECK: declare arm_aapcs_vfpcc void @returns_union_with_non_first_floats(%union.union_with_non_first_floats* sret)

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
// CHECK: declare arm_aapcs_vfpcc void @returns_struct_with_union_with_first_floats(%struct.struct_with_union_with_first_floats* sret)

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
// CHECK: declare arm_aapcs_vfpcc void @returns_struct_with_union_with_non_first_floats(%struct.struct_with_union_with_non_first_floats* sret)

/* Plain array is not a homogenous aggregate */
extern void takes_array_of_floats(float a[4]);
void test_array_of_floats(void) {
  float a[4] = {1.0, 2.0, 3.0, 4.0};
  takes_array_of_floats(a);
}
// CHECK: declare arm_aapcs_vfpcc void @takes_array_of_floats(float*)

/* Struct-type homogenous aggregate */
typedef struct {
  float x, y, z, w;
} struct_with_fundamental_elems;
struct_with_fundamental_elems g_s;

extern void takes_struct_with_fundamental_elems(struct_with_fundamental_elems a);
extern struct_with_fundamental_elems returns_struct_with_fundamental_elems(void);

void test_struct_with_fundamental_elems(void) {
  takes_struct_with_fundamental_elems(g_s);
// CHECK:  call arm_aapcs_vfpcc  void @takes_struct_with_fundamental_elems(float {{.*}}, float {{.*}}, float{{.*}}, float {{.*}})
}
// CHECK: declare arm_aapcs_vfpcc void @takes_struct_with_fundamental_elems(float, float, float, float)

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
// CHECK:   call arm_aapcs_vfpcc  void @takes_struct_with_array(float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}})
}
// CHECK: declare arm_aapcs_vfpcc void @takes_struct_with_array(float, float, float, float)

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
// CHECK: call arm_aapcs_vfpcc  void @takes_union_with_struct_with_fundamental_elems(float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}})
}
// CHECK: declare arm_aapcs_vfpcc void @takes_union_with_struct_with_fundamental_elems(float, float, float, float)

void test_return_union_with_struct_with_fundamental_elems(void) {
  g_u_s_fe = returns_union_with_struct_with_fundamental_elems();
// CHECK: call arm_aapcs_vfpcc  %union.union_with_struct_with_fundamental_elems @returns_union_with_struct_with_fundamental_elems()
}
// CHECK: declare arm_aapcs_vfpcc %union.union_with_struct_with_fundamental_elems @returns_union_with_struct_with_fundamental_elems()

// FIXME: Tests necessary:
//         - Vectors
//         - C++ stuff