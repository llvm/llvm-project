// REQUIRES: nvptx-registered-target
// REQUIRES: x86-registered-target

// RUN: %clang_cc1 "-aux-triple" "x86_64-unknown-linux-gnu" "-triple" "nvptx64-nvidia-cuda" \
// RUN:    -fcuda-is-device "-aux-target-cpu" "x86-64" -O1 -S -o - %s | FileCheck %s

#include "Inputs/cuda.h"

// CHECK-LABEL: .visible .func _Z8test_argPDF16bDF16b(
// CHECK:        .param .b64 _Z8test_argPDF16bDF16b_param_0,
// CHECK:        .param .align 2 .b8 _Z8test_argPDF16bDF16b_param_1[2]
//
__device__ void test_arg(__bf16 *out, __bf16 in) {
// CHECK-DAG:     ld.param.u64  %[[A:rd[0-9]+]], [_Z8test_argPDF16bDF16b_param_0];
// CHECK-DAG:     ld.param.b16  %[[R:rs[0-9]+]], [_Z8test_argPDF16bDF16b_param_1];
  __bf16 bf16 = in;
  *out = bf16;
// CHECK:         st.b16         [%[[A]]], %[[R]]
// CHECK:         ret;
}


// CHECK-LABEL: .visible .func (.param .align 2 .b8 func_retval0[2]) _Z8test_retDF16b(
// CHECK:        .param .align 2 .b8 _Z8test_retDF16b_param_0[2]
__device__ __bf16 test_ret( __bf16 in) {
// CHECK:        ld.param.b16    %[[R:rs[0-9]+]], [_Z8test_retDF16b_param_0];
  return in;
// CHECK:        st.param.b16    [func_retval0+0], %[[R]]
// CHECK:        ret;
}

__device__ __bf16 external_func( __bf16 in);

// CHECK-LABEL: .visible .func  (.param .align 2 .b8 func_retval0[2]) _Z9test_callDF16b(
// CHECK:        .param .align 2 .b8 _Z9test_callDF16b_param_0[2]
__device__ __bf16 test_call( __bf16 in) {
// CHECK:        ld.param.b16    %[[R:rs[0-9]+]], [_Z9test_callDF16b_param_0];
// CHECK:        st.param.b16    [param0+0], %[[R]];
// CHECK:        .param .align 2 .b8 retval0[2];
// CHECK:        call.uni (retval0),
// CHECK-NEXT:   _Z13external_funcDF16b,
// CHECK-NEXT:   (
// CHECK-NEXT:   param0
// CHECK-NEXT    );
// CHECK:        ld.param.b16    %[[RET:rs[0-9]+]], [retval0+0];
  return external_func(in);
// CHECK:        st.param.b16    [func_retval0+0], %[[RET]]
// CHECK:        ret;
}
