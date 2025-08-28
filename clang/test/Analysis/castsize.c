// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -Wno-alloc-size -analyzer-checker=core,unix.Malloc,alpha.core.CastSize

typedef typeof(sizeof(int)) size_t;
void *malloc(size_t);

struct s1 {
  int a;
  char x[];
};

struct s2 {
  int a[100];
  char x[];
};

union u {
  struct s1 a;
  struct s2 b;
};

static union u *test() {
  union u *req;
  req = malloc(5); // expected-warning{{Cast a region whose size is not a multiple of the destination type size}}
  return req;
}
