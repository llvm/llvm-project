
// GCN: {{^}}test_native_exp2_f32:
// GCN-NOT: v0
// GCN: v_exp_f32{{(_e32)?}} v0, v0
// GCN-NOT: v0
float test_native_exp2_f32(float arg) {
    return native_exp2(arg);
}

// GCN: {{^}}test_native_exp_f32:
// GCN-NOT: v0
// GCN: v_mul_f32{{(_e32)?}} v0, 0x3fb8aa3b, v0
// GCN-NEXT: v_exp_f32{{(_e32)?}} v0, v0
// GCN-NOT: v0
float test_native_exp_f32(float arg) {
    return native_exp(arg);
}

// GCN: {{^}}test_native_exp10_f32:
// GCN-NOT: v0
// GCN: v_mul_f32{{(_e32)?}} v0, 0x40549a78, v0
// GCN-NEXT: v_exp_f32{{(_e32)?}} v0, v0
// GCN-NOT: v0
float test_native_exp10_f32(float arg) {
    return native_exp10(arg);
}
