int main(int argc, char *argv[]) {
  static const void *T1[] = { &&L1, &&L2 };
  static const void *T2[] = { &&L2, &&L3 };

  const void **T = (argc > 1) ? T1 : T2;

  int i = 0;

L0:
  goto *T[argc];
L1:
  ++i;
L2:
  i++;
L3:
  i++;
  return i;
}
