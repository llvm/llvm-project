static const union C(codestrs_t_, NOW) {
  struct {
#define P(n, s) char MF(__LINE__)[sizeof (s)];
#include "psiginfo-data.h"
  };
  char str[0];
} C(codestrs_, NOW) = { {
#define P(n, s) s,
#include "psiginfo-data.h"
  } };
static const uint8_t C(codes_, NOW)[] = {
#define P(n, s) [(n) - 1] = offsetof (union C(codestrs_t_, NOW), MF(__LINE__)),
#include "psiginfo-data.h"
};
#undef NOW
