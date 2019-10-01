// RUN: %clang_cc1 %s -triple spir-unknown-unknown -finclude-default-header -verify -cl-std=CL1.2 -emit-llvm -o - -O0

void test_negative() {
    uchar4 ua8, ub8;
    char4 sa8, sb8;
    ushort2 ua16, ub16;
    short2 sa16, sb16;
    uint ur;
    int sr;
    ur = arm_dot(ua8, ub8); // expected-error{{no matching function for call to 'arm_dot'}}
    // expected-note@opencl-c.h:* {{candidate function not viable}}
    // expected-note@opencl-c.h:* {{candidate unavailable as it requires OpenCL extension 'cl_arm_integer_dot_product_int8' to be enabled}}
    sr = arm_dot(sa8, sb8); // expected-error{{no matching function for call to 'arm_dot'}}
    // expected-note@opencl-c.h:* {{candidate function not viable}}
    // expected-note@opencl-c.h:* {{candidate unavailable as it requires OpenCL extension 'cl_arm_integer_dot_product_int8' to be enabled}}
    ur = arm_dot_acc(ua8, ub8, ur); // expected-error{{no matching function for call to 'arm_dot_acc'}}
    // expected-note@opencl-c.h:* {{candidate function not viable}}
    // expected-note@opencl-c.h:* {{candidate function not viable}}
    // expected-note@opencl-c.h:* {{candidate function not viable}}
    // expected-note@opencl-c.h:* {{candidate unavailable as it requires OpenCL extension 'cl_arm_integer_dot_product_accumulate_int8' to be enabled}}
    sr = arm_dot_acc(sa8, sb8, sr); // expected-error{{no matching function for call to 'arm_dot_acc'}}
    // expected-note@opencl-c.h:* {{candidate function not viable}}
    // expected-note@opencl-c.h:* {{candidate function not viable}}
    // expected-note@opencl-c.h:* {{candidate function not viable}}
    // expected-note@opencl-c.h:* {{candidate unavailable as it requires OpenCL extension 'cl_arm_integer_dot_product_accumulate_int8' to be enabled}}
    ur = arm_dot_acc(ua16, ub16, ur); // expected-error{{no matching function for call to 'arm_dot_acc'}}
    // expected-note@opencl-c.h:* {{candidate function not viable}}
    // expected-note@opencl-c.h:* {{candidate function not viable}}
    // expected-note@opencl-c.h:* {{candidate function not viable}}
    // expected-note@opencl-c.h:* {{candidate unavailable as it requires OpenCL extension 'cl_arm_integer_dot_product_accumulate_int16' to be enabled}}
    sr = arm_dot_acc(sa16, sb16, sr); // expected-error{{no matching function for call to 'arm_dot_acc'}}
    // expected-note@opencl-c.h:* {{candidate function not viable}}
    // expected-note@opencl-c.h:* {{candidate function not viable}}
    // expected-note@opencl-c.h:* {{candidate function not viable}}
    // expected-note@opencl-c.h:* {{candidate unavailable as it requires OpenCL extension 'cl_arm_integer_dot_product_accumulate_int16' to be enabled}}
    ur = arm_dot_acc_sat(ua8, ub8, ur); // expected-error{{no matching function for call to 'arm_dot_acc_sat'}}
    // expected-note@opencl-c.h:* {{candidate function not viable}}
    // expected-note@opencl-c.h:* {{candidate unavailable as it requires OpenCL extension 'cl_arm_integer_dot_product_accumulate_saturate_int8' to be enabled}}
    sr = arm_dot_acc_sat(sa8, sb8, sr); // expected-error{{no matching function for call to 'arm_dot_acc_sat'}}
    // expected-note@opencl-c.h:* {{candidate function not viable}}
    // expected-note@opencl-c.h:* {{candidate unavailable as it requires OpenCL extension 'cl_arm_integer_dot_product_accumulate_saturate_int8' to be enabled}}
}

