__kernel void atomic_kernel(__global uint *out, __global uint *counter) {
    uint gid = get_global_id(0);
    uint old = atomic_add(counter, 1u);
    out[gid] = old;
}
