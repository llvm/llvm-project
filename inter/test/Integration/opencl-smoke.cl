// REQUIRES: host-supports-inter-bmg
// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir
// RUN: ocloc compile -file %s -device bmg-g21 -out_dir %t.dir
// RUN: inter-runner %t.dir/opencl-smoke_bmg.bin opencl_smoke 128 out u32:7 | %python %S/../../verify.py 'i*2+7'

__kernel void opencl_smoke(__global unsigned int *out, unsigned int bias) {
  size_t gid = get_global_id(0);
  out[gid] = (unsigned int)gid * 2u + bias;
}
