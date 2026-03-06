// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -emit-llvm -o - %s | FileCheck %s

// Check that structs containing mixed float and int types are not coerced
// to integer arrays. They should preserve the original struct type and
// individual field types.

typedef struct fp_int_pair {
    float f;
    int i;
} fp_int_pair;

// CHECK-LABEL: define{{.*}} %struct.fp_int_pair @return_fp_int_pair(float %x.coerce0, i32 %x.coerce1)
// CHECK: ret %struct.fp_int_pair
fp_int_pair return_fp_int_pair(fp_int_pair x) {
    return x;
}

typedef struct int_fp_pair {
    int i;
    float f;
} int_fp_pair;

// CHECK-LABEL: define{{.*}} %struct.int_fp_pair @return_int_fp_pair(i32 %x.coerce0, float %x.coerce1)
// CHECK: ret %struct.int_fp_pair
int_fp_pair return_int_fp_pair(int_fp_pair x) {
    return x;
}

typedef struct two_floats {
    float a;
    float b;
} two_floats;

// CHECK-LABEL: define{{.*}} %struct.two_floats @return_two_floats(float %x.coerce0, float %x.coerce1)
// CHECK: ret %struct.two_floats
two_floats return_two_floats(two_floats x) {
    return x;
}

typedef struct two_ints {
    int a;
    int b;
} two_ints;

// CHECK-LABEL: define{{.*}} %struct.two_ints @return_two_ints(i32 %x.coerce0, i32 %x.coerce1)
// CHECK: ret %struct.two_ints
two_ints return_two_ints(two_ints x) {
    return x;
}

// Structs <= 32 bits should still be coerced to i32 for return value
typedef struct small_struct {
    short a;
    short b;
} small_struct;

// CHECK-LABEL: define{{.*}} i32 @return_small_struct(i16 %x.coerce0, i16 %x.coerce1)
small_struct return_small_struct(small_struct x) {
    return x;
}

// Structs <= 16 bits should still be coerced to i16 for return value
typedef struct tiny_struct {
    char a;
    char b;
} tiny_struct;

// CHECK-LABEL: define{{.*}} i16 @return_tiny_struct(i8 %x.coerce0, i8 %x.coerce1)
tiny_struct return_tiny_struct(tiny_struct x) {
    return x;
}
