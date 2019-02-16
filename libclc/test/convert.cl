__kernel void foo(int4 *x, float4 *y) {
  *x = convert_int4(*y);
}
