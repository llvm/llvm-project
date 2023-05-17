// REQUIRES: nvptx-registered-target
// REQUIRES: x86-registered-target

// RUN: %clang_cc1 "-aux-triple" "x86_64-unknown-linux-gnu" "-triple" "nvptx64-nvidia-cuda" \
// RUN:    -fcuda-is-device "-aux-target-cpu" "x86-64" -S -o - %s | FileCheck %s

#include "Inputs/cuda.h"

// CHECK-LABEL: .visible .func _Z8test_argPu6__bf16u6__bf16(
// CHECK:        .param .b64 _Z8test_argPu6__bf16u6__bf16_param_0,
// CHECK:        .param .b16 _Z8test_argPu6__bf16u6__bf16_param_1
//
__device__ void test_arg(__bf16 *out, __bf16 in) {
// CHECK:         ld.param.b16    %{{h.*}}, [_Z8test_argPu6__bf16u6__bf16_param_1];
  __bf16 bf16 = in;
  *out = bf16;
// CHECK:         st.b16
// CHECK:         ret;
}


// CHECK-LABEL: .visible .func (.param .b32 func_retval0) _Z8test_retu6__bf16(
// CHECK:         .param .b16 _Z8test_retu6__bf16_param_0
__device__ __bf16 test_ret( __bf16 in) {
// CHECK:        ld.param.b16    %h{{.*}}, [_Z8test_retu6__bf16_param_0];
  return in;
// CHECK:        st.param.b16    [func_retval0+0], %h
// CHECK:        ret;
}

// CHECK-LABEL: .visible .func  (.param .b32 func_retval0) _Z9test_callu6__bf16(
// CHECK:        .param .b16 _Z9test_callu6__bf16_param_0
__device__ __bf16 test_call( __bf16 in) {
// CHECK:        ld.param.b16    %h{{.*}}, [_Z9test_callu6__bf16_param_0];
// CHECK:        st.param.b16    [param0+0], %h2;
// CHECK:        .param .b32 retval0;
// CHECK:        call.uni (retval0),
// CHECK-NEXT:   _Z8test_retu6__bf16,
// CHECK-NEXT:   (
// CHECK-NEXT:   param0
// CHECK-NEXT    );
// CHECK:        ld.param.b16    %h{{.*}}, [retval0+0];
  return test_ret(in);
// CHECK:        st.param.b16    [func_retval0+0], %h
// CHECK:        ret;
}
