
// Use an internal, implicitly defined type, called by
// a function imported for CTU. This should not crash.
int foo(void);
int foobar(int skip) {
  __NSConstantString str = {.flags = 1};

  if (str.flags >= 0)
    str.flags = 0;
  return 4;
}

int testStaticImplicit(void) {
  return foobar(3);
}
