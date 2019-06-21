// REQUIRES: x86-registered-target
INLINE int bar() {
  static int var = 42;
  return var;
}
