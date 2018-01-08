__kernel void test_subsat_char(char *a, char x, char y) {
  *a = sub_sat(x, y);
  return;
}

__kernel void test_subsat_uchar(uchar *a, uchar x, uchar y) {
  *a = sub_sat(x, y);
  return;
}

__kernel void test_subsat_long(long *a, long x, long y) {
  *a = sub_sat(x, y);
  return;
}

__kernel void test_subsat_ulong(ulong *a, ulong x, ulong y) {
  *a = sub_sat(x, y);
  return;
}