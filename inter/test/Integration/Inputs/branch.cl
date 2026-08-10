__kernel void branch_kernel(__global uint *out, __global const uint *a, __global const uint *b, uint t) {
    uint gid = get_global_id(0);
    uint v = a[gid];
    if (v > t) {
        v = v + b[gid];
    } else {
        v = v + 1u;
    }
    out[gid] = v;
}
