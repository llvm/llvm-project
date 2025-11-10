// REQUIRES: target=hexagon{{.*}} || target-x86_64
// RUN: %clang -S -O2 -emit-llvm %s -DCOMPILE_LIB=1 -o %t.ll
// RUN: %clang -S -O2 -emit-llvm -fenable-ripple -fripple-lib=%t.ll -mllvm -ripple-disable-link %s &> %t.warnings && FileCheck %s --input-file=%t.warnings

#ifdef COMPILE_LIB

typedef float v4f32 __attribute__((__vector_size__(16)))
__attribute__((aligned(16)));
typedef float v16f32 __attribute__((__vector_size__(64)))
__attribute__((aligned(16)));

// Missing tensor type
// CHECK: warning: the external ripple function 'ripple_uniform_aa' uniform shape requires a tensor shape specifier between 'uniform_' and 'aa', e.g., 'uniform_t2x4f32_aa'
extern v16f32 ripple_uniform_aa(v16f32 A, v16f32 B) { return A + B; }

// Missing tensor type and name
// CHECK: warning: the external ripple function 'ripple_uniform_' uniform shape requires a tensor shape specifier and a function name, e.g., 'uniform_t2x4f32_myFunction'
// CHECK: warning: the external ripple function 'ripple_uniform_' is missing a function name following the function options and tensor shapes, e.g., 'ripple_uniform_myFunction'
extern v16f32 ripple_uniform_(v16f32 A, v16f32 B) { return A + B; }

// Malformed tensor shape
// CHECK: warning: the external ripple function 'ripple_ret_tf32_bb' 'ret_' tensor shape is invalid 'tf32_bb'; expected 't<dims><type>' (e.g., 't2x4f32')
extern v16f32 ripple_ret_tf32_bb(v16f32 A, v16f32 B) { return A + B; }

// Malformed tensor shape
// warning: the external ripple function 'ripple_arg1_tf32_bb' 'arg1_' tensor shape is invalid 'tf32_bb'; expected 't<dims><type>' (e.g., 't2x4f32')
extern v16f32 ripple_arg1_tf32_bb(v16f32 A, v16f32 B) { return A + B; }

// Missing _ after tensor shape
// CHECK: warning: the external ripple function 'ripple_ew_t4x4f32bb' tensor shape expects '_' after 't4x4f32' before 'bb'
extern v16f32 ripple_ew_t4x4f32bb(v16f32 A, v16f32 B) { return A + B; }

// Missing _ after tensor shape and name
// CHECK: warning: the external ripple function 'ripple_ew_t4x4f32' tensor shape expects '_' after 't4x4f32'
extern v16f32 ripple_ew_t4x4f32(v16f32 A, v16f32 B) { return A + B; }

// Mixing uniform and ret
// CHECK: warning: in the external ripple function 'ripple_ew_uniform_ret_t16x2f32_aa' 'uniform_' cannot be combined with 'ret_'; use one or the other
extern v16f32 ripple_ew_uniform_ret_t16x2f32_aa(v16f32 A, v16f32 B) { return A + B; }

// Mixing uniform and arg
// CHECK: warning: in the external ripple function 'ripple_ew_uniform_arg0_t32f32_aa' 'uniform_' cannot be combined with 'arg0_'; use one or the other
extern v16f32 ripple_ew_uniform_arg0_t32f32_aa(v16f32 A, v16f32 B) { return A + B; }

// Missing arg index
// CHECK: warning: the external ripple function 'ripple_ew_uniform_arg_t32f32_aa' 'arg' tensor shape requires an index followed by an underscore (e.g., 'arg1_t2x4f32'); 'arg_t32f32_aa' is invalid
// CHECK: warning: the external ripple function 'ripple_ew_uniform_arg_t32f32_aa' uniform shape requires a tensor shape specifier between 'uniform_' and 'arg_t32f32_aa', e.g., 'uniform_t2x4f32_arg_t32f32_aa'
extern v16f32 ripple_ew_uniform_arg_t32f32_aa(v16f32 A, v16f32 B) { return A + B; }

// Unknown datatype f400
// CHECK: warning: the external ripple function 'ripple_ew_uniform_t16x2f400_aa' tensor shape's type is not valid starting at 'f400_aa'; expected one of: f16, bf16, f32, f64, i1, i8, i16, i32, i64, u8, u16, u32 or u64
// CHECK: warning: the external ripple function 'ripple_ew_uniform_t16x2f400_aa' uniform shape requires a tensor shape specifier between 'uniform_' and 't16x2f400_aa', e.g., 'uniform_t2x4f32_t16x2f400_aa'
extern v16f32 ripple_ew_uniform_t16x2f400_aa(v16f32 A, v16f32 B) { return A + B; }

// No function name
// CHECK: warning: the external ripple function 'ripple_ew_uniform_t4x4f32_' is missing a function name following the function options and tensor shapes, e.g., 'ripple_ew_uniform_t4x4f32_myFunction'
extern v16f32 ripple_ew_uniform_t4x4f32_(v16f32 A, v16f32 B) { return A + B; }

// Arg0 shape defined twice
// CHECK: warning: the external ripple function 'ripple_t16f32_arg0_t16f32_bb' has duplicate tensor shape specified for 'arg0'; second definition starts at 'arg0_t16f32_bb'
extern v16f32 ripple_t16f32_arg0_t16f32_bb(v16f32 A, v16f32 B) { return A + B; }

// Return shape defined twice
// CHECK: warning: the external ripple function 'ripple_ret_t16f32_arg0_t16f32_ret_t16f32_bb' has duplicate tensor shape specified for 'ret'; second definition starts at 'ret_t16f32_bb'
extern v16f32 ripple_ret_t16f32_arg0_t16f32_ret_t16f32_bb(v16f32 A, v16f32 B) { return A + B; }

// Argument index OOB
// CHECK: warning: the external ripple function 'ripple_ret_t16f32_arg9_t16f32_bb' tensor shape argument index 9 is out of bound; function has only 2 arguments
extern v16f32 ripple_ret_t16f32_arg9_t16f32_bb(v16f32 A, v16f32 B) { return A + B; }

// Wrong argument tensor shape
// CHECK: warning: the external ripple function 'ripple_ret_t16f32_arg1_t16x2f32_cc' argument at index 1 tensor shape 'Tensor[16][2]' is incompatible with a vector size of 16
extern v16f32 ripple_ret_t16f32_arg1_t16x2f32_cc(v16f32 A, v16f32 B) { return A + B; }

// Wrong return tensor shape
// CHECK: warning: the external ripple function 'ripple_ret_t8x4f32_arg1_t4x4f32_cc' return value tensor shape 'Tensor[8][4]' is incompatible with a vector size of 16
extern v16f32 ripple_ret_t8x4f32_arg1_t4x4f32_cc(v16f32 A, v16f32 B) { return A + B; }

#else

#include <ripple.h>

void test_valid_candidates(float *restrict f, float *restrict f2, float *restrict f3) {
  ripple_block_t BS = ripple_set_block_shape(0, 32, 32);
  size_t rid = ripple_id(BS, 0);
  f[rid] = f2[rid] + f3[rid];
}

#endif // COMPILE_LIB
