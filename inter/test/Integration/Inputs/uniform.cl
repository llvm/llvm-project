__kernel void uniform_kernel(__global uint *out, __global const uint *a, __global const uint *b, uint t) {
    uint gid = get_global_id(0);
    if (t > 3u) {
        out[gid] = a[gid] + 100u;
    } else {
        out[gid] = b[gid];
    }
}
