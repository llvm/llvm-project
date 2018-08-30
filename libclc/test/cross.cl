__kernel void foo(float4 *f) {
  *f = cross(f[0], f[1]);
}
