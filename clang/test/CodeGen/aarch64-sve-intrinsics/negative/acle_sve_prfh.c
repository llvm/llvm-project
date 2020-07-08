// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -fsyntax-only -verify %s

#include <arm_sve.h>

void test_svprfh(svbool_t pg, const void *base)
{
  // expected-error@+1 {{argument value 14 is outside the valid range [0, 13]}}
  return svprfh(pg, base, 14);
}

void test_svprfh_1(svbool_t pg, const void *base)
{
  // expected-error@+1 {{argument value -1 is outside the valid range [0, 13]}}
  return svprfh(pg, base, -1);
}

void test_svprfh_vnum(svbool_t pg, const void *base)
{
  // expected-error@+1 {{argument value 14 is outside the valid range [0, 13]}}
  return svprfh_vnum(pg, base, 0, 14);
}

void test_svprfh_vnum_1(svbool_t pg, const void *base)
{
  // expected-error@+1 {{argument value -1 is outside the valid range [0, 13]}}
  return svprfh_vnum(pg, base, 0, -1);
}
