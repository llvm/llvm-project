__kernel void foo(int4 *x, float4 *y) {
  *x = as_int4(*y);
}
