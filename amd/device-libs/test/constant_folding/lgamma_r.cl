// Verify lgamma_r function constant folds to correct values.
// Run with filecheck from test cmake

__attribute__((always_inline))
static float test_lgamma_r(float val, volatile global int* sign_out) {
   int tmp;
   float result = lgamma_r(val, &tmp);
   *sign_out = tmp;
   return result;
}

// CHECK-LABEL: @constant_fold_lgamma_r_f32(
kernel void constant_fold_lgamma_r_f32(volatile global float* out,
                                       volatile global int* sign_out) {
    // CHECK: store volatile i32 0,
    // CHECK-NEXT: store volatile float 0x7FF0000000000000
    out[0] = test_lgamma_r(0.0f, sign_out);

    // CHECK-NEXT: store volatile i32 0,
    // CHECK-NEXT: store volatile float 0x7FF0000000000000
    out[0] = test_lgamma_r(-0.0f, sign_out);

    // CHECK-NEXT: store volatile i32 0,
    // CHECK-NEXT: store volatile float 0x7FF8000000000000,
    out[0] = test_lgamma_r(__builtin_nanf(""), sign_out);

    // CHECK-NEXT: store volatile i32 0,
    // CHECK-NEXT: store volatile float 0x7FF4000000000000,
    out[0] = test_lgamma_r(__builtin_nansf(""), sign_out);

    // CHECK-NEXT: store volatile i32 1,
    // CHECK-NEXT: store volatile float 0x7FF0000000000000,
    out[0] = test_lgamma_r(__builtin_inff(), sign_out);

    // CHECK-NEXT: store volatile i32 0,
    // CHECK-NEXT: store volatile float 0x7FF0000000000000,
    out[0] = test_lgamma_r(-__builtin_inff(), sign_out);

    // CHECK-NEXT: store volatile i32 1,
    // CHECK-NEXT: store volatile float 0x419DE28020000000,
    out[0] = test_lgamma_r(0x1.0p+23f, sign_out);

    // CHECK-NEXT: store volatile i32 0,
    // CHECK-NEXT: store volatile float 0x7FF0000000000000,
    out[0] = test_lgamma_r(-0x1.0p+23f, sign_out);

    // CHECK-NEXT: store volatile i32 1,
    // CHECK-NEXT: store volatile float 0.000000e+00,
    out[0] = test_lgamma_r(1.0f, sign_out);

    // CHECK-NEXT: store volatile i32 1,
    // CHECK-NEXT: store volatile float 0.000000e+00,
    out[0] = test_lgamma_r(2.0f, sign_out);

    // CHECK-NEXT: store volatile i32 1,
    // CHECK-NEXT: store volatile float 0x3FE62E4300000000,
    out[0] = test_lgamma_r(3.0f, sign_out);

    // CHECK-NEXT: store volatile i32 1,
    // CHECK-NEXT: store volatile float 0x3FE250D040000000,
    out[0] = test_lgamma_r(0.5f, sign_out);

    // CHECK-NEXT: store volatile i32 1,
    // CHECK-NEXT: store volatile float 0x405601E680000000,
    out[0] = test_lgamma_r(0x1.0p-127f, sign_out);
}
