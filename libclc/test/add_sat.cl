__kernel void foo(__global char *a, __global char *b, __global char *c) {
  *a = add_sat(*b, *c);
}
