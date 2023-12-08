// RUN: clang-tidy -checks=-*,google-runtime-int %s -- -x c 2>&1 | not grep 'warning:\|error:'

long a(void);

long b(long x);

short bar(const short q, unsigned short w) {
  long double foo;
  unsigned short port;

  const unsigned short bar;
  long long *baar;
  const unsigned short bara;
  long const long moo;
  long volatile long wat;
  unsigned long y;
  unsigned long long **const *tmp;
  unsigned short porthole;

  unsigned cast;
  cast = (short)42;
  return q;
}

void qux(void) {
  short port;
}
