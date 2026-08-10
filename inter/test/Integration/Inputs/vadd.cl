__kernel void vadd(__global const unsigned int *a, __global const unsigned int *b,
                   __global unsigned int *out) {
    size_t gid = get_global_id(0);
    out[gid] = a[gid] + b[gid];
}
