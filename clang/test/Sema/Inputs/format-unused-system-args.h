// "System header" for testing that -Wformat-extra-args does not apply to
// arguments specified in system headers.

#define PRINT2(fmt, a1, a2) \
  printf((fmt), (a1), (a2))

#define PRINT1(fmt, a1) \
  PRINT2((fmt), (a1), 0)
