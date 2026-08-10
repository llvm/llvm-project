__kernel void slm_kernel(__global uint *out, __global const uint *in) {
    __local uint tile[32];
    uint lid = get_local_id(0);
    uint gid = get_global_id(0);
    tile[lid] = in[gid];
    barrier(CLK_LOCAL_MEM_FENCE);
    out[gid] = tile[31u - lid] + tile[lid];
}
