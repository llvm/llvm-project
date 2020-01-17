int main() {
  int res;
  __asm__("add %[r], r0, 1\n" : [ r ] "=r"(res));
  return res;
}
