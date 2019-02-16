// RUN: %clang_analyze_cc1 -analyzer-store=region -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix \
// RUN:   -analyzer-checker=core.uninitialized \
// RUN:   -analyzer-config unix:Optimistic=true

typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);
void free(void *);

char stackBased1 () {
  char buf[2];
  buf[0] = 'a';
  return buf[1]; // expected-warning{{Undefined}}
}

char stackBased2 () {
  char buf[2];
  buf[1] = 'a';
  return buf[0]; // expected-warning{{Undefined}}
}

// Exercise the conditional visitor. Radar://10105448
char stackBased3 (int *x) {
  char buf[2];
  int *y;
  buf[0] = 'a';
  if (!(y = x)) {
    return buf[1]; // expected-warning{{Undefined}}
  }
  return buf[0];
}

char heapBased1 () {
  char *buf = malloc(2);
  buf[0] = 'a';
  char result = buf[1]; // expected-warning{{undefined}}
  free(buf);
  return result;
}

char heapBased2 () {
  char *buf = malloc(2);
  buf[1] = 'a';
  char result = buf[0]; // expected-warning{{undefined}}
  free(buf);
  return result;
}
