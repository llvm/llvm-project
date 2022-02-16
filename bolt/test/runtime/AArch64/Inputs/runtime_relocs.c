int a = 1;
__attribute__((used)) int *b = &a; // R_*_ABS64

static int c;
__attribute__((used)) static int *d = &c; // R_*_RELATIVE

__thread int t1 = 0;

int inc(int var) {
  ++a;  // R_*_GLOB_DAT
  ++t1; // R_*_TLSDESC
  return var + 1;
}
