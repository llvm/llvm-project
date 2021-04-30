// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify %s

#include <arm_sve.h>

void test_svprfd(svbool_t pg, const void *base)
{
  // expected-error@+1 {{argument value 14 is outside the valid range [0, 13]}}
  return svprfd(pg, base, 14);
}

void test_svprfd_1(svbool_t pg, const void *base)
{
  // expected-error-re@+1 {{argument value {{.*}} is outside the valid range [0, 13]}}
  return svprfd(pg, base, -1);
}

void test_svprfd_vnum(svbool_t pg, const void *base)
{
  // expected-error@+1 {{argument value 14 is outside the valid range [0, 13]}}
  return svprfd_vnum(pg, base, 0, 14);
}

void test_svprfd_vnum_1(svbool_t pg, const void *base)
{
  // expected-error-re@+1 {{argument value {{.*}} is outside the valid range [0, 13]}}
  return svprfd_vnum(pg, base, 0, -1);
}
