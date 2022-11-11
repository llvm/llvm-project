// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve2 -fsyntax-only -verify -verify-ignore-unexpected=error,note %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve2 -fsyntax-only -verify=overload -verify-ignore-unexpected=error,note %s

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

#include <arm_sve.h>

void test_u8(uint8_t u8)
{
  // expected-error@+2 {{use of undeclared identifier 'svaesd_u8'}}
  // overload-error@+1 {{use of undeclared identifier 'svaesd'}}
  SVE_ACLE_FUNC(svaesd,_u8,,)(svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svaese_u8'}}
  // overload-error@+1 {{use of undeclared identifier 'svaese'}}
  SVE_ACLE_FUNC(svaese,_u8,,)(svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svaesimc_u8'}}
  // overload-error@+1 {{use of undeclared identifier 'svaesimc'}}
  SVE_ACLE_FUNC(svaesimc,_u8,,)(svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svaesmc_u8'}}
  // overload-error@+1 {{use of undeclared identifier 'svaesmc'}}
  SVE_ACLE_FUNC(svaesmc,_u8,,)(svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svbdep_u8'}}
  // overload-error@+1 {{use of undeclared identifier 'svbdep'}}
  SVE_ACLE_FUNC(svbdep,_u8,,)(svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svbdep_n_u8'}}
  // overload-error@+1 {{use of undeclared identifier 'svbdep'}}
  SVE_ACLE_FUNC(svbdep,_n_u8,,)(svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svbext_u8'}}
  // overload-error@+1 {{use of undeclared identifier 'svbext'}}
  SVE_ACLE_FUNC(svbext,_u8,,)(svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svbext_n_u8'}}
  // overload-error@+1 {{use of undeclared identifier 'svbext'}}
  SVE_ACLE_FUNC(svbext,_n_u8,,)(svundef_u8(), u8);
  // expected-error@+2 {{use of undeclared identifier 'svbgrp_u8'}}
  // overload-error@+1 {{use of undeclared identifier 'svbgrp'}}
  SVE_ACLE_FUNC(svbgrp,_u8,,)(svundef_u8(), svundef_u8());
  // expected-error@+2 {{use of undeclared identifier 'svbgrp_n_u8'}}
  // overload-error@+1 {{use of undeclared identifier 'svbgrp'}}
  SVE_ACLE_FUNC(svbgrp,_n_u8,,)(svundef_u8(), u8);
}

void test_u16(uint16_t u16)
{
  // expected-error@+2 {{use of undeclared identifier 'svbdep_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svbdep'}}
  SVE_ACLE_FUNC(svbdep,_u16,,)(svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svbdep_n_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svbdep'}}
  SVE_ACLE_FUNC(svbdep,_n_u16,,)(svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svbext_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svbext'}}
  SVE_ACLE_FUNC(svbext,_u16,,)(svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svbext_n_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svbext'}}
  SVE_ACLE_FUNC(svbext,_n_u16,,)(svundef_u16(), u16);
  // expected-error@+2 {{use of undeclared identifier 'svbgrp_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svbgrp'}}
  SVE_ACLE_FUNC(svbgrp,_u16,,)(svundef_u16(), svundef_u16());
  // expected-error@+2 {{use of undeclared identifier 'svbgrp_n_u16'}}
  // overload-error@+1 {{use of undeclared identifier 'svbgrp'}}
  SVE_ACLE_FUNC(svbgrp,_n_u16,,)(svundef_u16(), u16);
}

void test_u32(uint32_t u32)
{
  // expected-error@+2 {{use of undeclared identifier 'svbdep_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svbdep'}}
  SVE_ACLE_FUNC(svbdep,_u32,,)(svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svbdep_n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svbdep'}}
  SVE_ACLE_FUNC(svbdep,_n_u32,,)(svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svbext_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svbext'}}
  SVE_ACLE_FUNC(svbext,_u32,,)(svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svbext_n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svbext'}}
  SVE_ACLE_FUNC(svbext,_n_u32,,)(svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svbgrp_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svbgrp'}}
  SVE_ACLE_FUNC(svbgrp,_u32,,)(svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svbgrp_n_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svbgrp'}}
  SVE_ACLE_FUNC(svbgrp,_n_u32,,)(svundef_u32(), u32);
  // expected-error@+2 {{use of undeclared identifier 'svsm4e_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svsm4e'}}
  SVE_ACLE_FUNC(svsm4e,_u32,,)(svundef_u32(), svundef_u32());
  // expected-error@+2 {{use of undeclared identifier 'svsm4ekey_u32'}}
  // overload-error@+1 {{use of undeclared identifier 'svsm4ekey'}}
  SVE_ACLE_FUNC(svsm4ekey,_u32,,)(svundef_u32(), svundef_u32());
}

void test_u64(uint64_t u64)
{
  // expected-error@+2 {{use of undeclared identifier 'svbdep_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svbdep'}}
  SVE_ACLE_FUNC(svbdep,_u64,,)(svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svbdep_n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svbdep'}}
  SVE_ACLE_FUNC(svbdep,_n_u64,,)(svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svbext_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svbext'}}
  SVE_ACLE_FUNC(svbext,_u64,,)(svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svbext_n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svbext'}}
  SVE_ACLE_FUNC(svbext,_n_u64,,)(svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svbgrp_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svbgrp'}}
  SVE_ACLE_FUNC(svbgrp,_u64,,)(svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svbgrp_n_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svbgrp'}}
  SVE_ACLE_FUNC(svbgrp,_n_u64,,)(svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svpmullb_pair_u64'}}
  // overload-error@+1 {{no matching function for call to 'svpmullb_pair'}}
  SVE_ACLE_FUNC(svpmullb_pair,_u64,,)(svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svpmullb_pair_n_u64'}}
  // overload-error@+1 {{no matching function for call to 'svpmullb_pair'}}
  SVE_ACLE_FUNC(svpmullb_pair,_n_u64,,)(svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svpmullt_pair_u64'}}
  // overload-error@+1 {{no matching function for call to 'svpmullt_pair'}}
  SVE_ACLE_FUNC(svpmullt_pair,_u64,,)(svundef_u64(), svundef_u64());
  // expected-error@+2 {{use of undeclared identifier 'svpmullt_pair_n_u64'}}
  // overload-error@+1 {{no matching function for call to 'svpmullt_pair'}}
  SVE_ACLE_FUNC(svpmullt_pair,_n_u64,,)(svundef_u64(), u64);
  // expected-error@+2 {{use of undeclared identifier 'svrax1_u64'}}
  // overload-error@+1 {{use of undeclared identifier 'svrax1'}}
  SVE_ACLE_FUNC(svrax1,_u64,,)(svundef_u64(), svundef_u64());
}

void test_s64()
{
  // expected-error@+2 {{use of undeclared identifier 'svrax1_s64'}}
  // overload-error@+1 {{use of undeclared identifier 'svrax1'}}
  SVE_ACLE_FUNC(svrax1,_s64,,)(svundef_s64(), svundef_s64());
}
