// REQUIRES: x86-registered-target
INLINE int bar(void) {
  static int var = 42;
  return var;
}
