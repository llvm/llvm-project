__kernel void scale(__global unsigned int *out, unsigned int bias) {
    size_t gid = get_global_id(0);
    out[gid] = (unsigned int)gid * 2u + bias;
}
