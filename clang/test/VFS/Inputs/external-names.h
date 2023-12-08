void foo(char **c) {
  *c = __FILE__;
  const char **x = c; // produce a diagnostic
}
