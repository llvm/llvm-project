// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.core.SizeofPtr -verify %s

struct s {
};

int f(struct s *p) {
  return sizeof(p); // expected-warning{{The code calls sizeof() on a pointer type. This can produce an unexpected result}}
}
