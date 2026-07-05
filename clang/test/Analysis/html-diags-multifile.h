#define DEREF(p) *p = 0xDEADBEEF
void has_bug(int *p) {
  DEREF(p);
}
