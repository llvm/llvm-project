
// GCN: {{^}}test_native_log_f32:
// GCN-NOT: v0
// GCN: v_log_f32{{(_e32)?}} v0, v0
// GCN-NEXT: v_mul_f32{{(_e32)?}} v0, 0x3f317218, v0
// GCN-NOT: v0
float test_native_log_f32(float arg) {
    return native_log(arg);
}

// GCN: {{^}}test_native_log2_f32:
// GCN-NOT: v0
// GCN: v_log_f32{{(_e32)?}} v0, v0
// GCN-NOT: v0
float test_native_log2_f32(float arg) {
    return native_log2(arg);
}

// GCN: {{^}}test_native_log10_f32:
// GCN-NOT: v0
// GCN: v_log_f32{{(_e32)?}} v0, v0
// GCN-NEXT: v_mul_f32{{(_e32)?}} v0, 0x3e9a209b, v0

// GCN-NOT: v0
float test_native_log10_f32(float arg) {
    return native_log10(arg);
}
