// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

typedef float __attribute__((vector_size (16))) v4f_t;

typedef union {
    struct {
        float x, y, z, w;
    }s;
    v4f_t v;
} vector_t;


vector_t foo(v4f_t p)
{
  vector_t v = {.v = p};
  return v;
}
