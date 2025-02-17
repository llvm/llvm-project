// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +sve2 -verify -verify-ignore-unexpected=error,note -emit-llvm -o - %s
// RUN: %clang_cc1 -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +sve2 -verify=overload -verify-ignore-unexpected=error,note -emit-llvm -o - %s

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

#include <arm_sve.h>

void test(uint8_t u8, uint16_t u16, uint32_t u32, uint64_t u64)
{
  // expected-error@+2 {{'svaesd_u8' needs target feature sve,sve2,sve-aes}}
  // overload-error@+1 {{'svaesd' needs target feature sve,sve2,sve-aes}}
  SVE_ACLE_FUNC(svaesd,_u8,,)(svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svaese_u8' needs target feature sve,sve2,sve-aes}}
  // overload-error@+1 {{'svaese' needs target feature sve,sve2,sve-aes}}
  SVE_ACLE_FUNC(svaese,_u8,,)(svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svaesimc_u8' needs target feature sve,sve2,sve-aes}}
  // overload-error@+1 {{'svaesimc' needs target feature sve,sve2,sve-aes}}
  SVE_ACLE_FUNC(svaesimc,_u8,,)(svundef_u8());
  // expected-error@+2 {{'svaesmc_u8' needs target feature sve,sve2,sve-aes}}
  // overload-error@+1 {{'svaesmc' needs target feature sve,sve2,sve-aes}}
  SVE_ACLE_FUNC(svaesmc,_u8,,)(svundef_u8());
  // expected-error@+2 {{'svbdep_u8' needs target feature sve,sve2,sve-bitperm}}
  // overload-error@+1 {{'svbdep' needs target feature sve,sve2,sve-bitperm}}
  SVE_ACLE_FUNC(svbdep,_u8,,)(svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svbdep_n_u8' needs target feature sve,sve2,sve-bitperm}}
  // overload-error@+1 {{'svbdep' needs target feature sve,sve2,sve-bitperm}}
  SVE_ACLE_FUNC(svbdep,_n_u8,,)(svundef_u8(), u8);
  // expected-error@+2 {{'svbext_u8' needs target feature sve,sve2,sve-bitperm}}
  // overload-error@+1 {{'svbext' needs target feature sve,sve2,sve-bitperm}}
  SVE_ACLE_FUNC(svbext,_u8,,)(svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svbext_n_u8' needs target feature sve,sve2,sve-bitperm}}
  // overload-error@+1 {{'svbext' needs target feature sve,sve2,sve-bitperm}}
  SVE_ACLE_FUNC(svbext,_n_u8,,)(svundef_u8(), u8);
  // expected-error@+2 {{'svbgrp_u8' needs target feature sve,sve2,sve-bitperm}}
  // overload-error@+1 {{'svbgrp' needs target feature sve,sve2,sve-bitperm}}
  SVE_ACLE_FUNC(svbgrp,_u8,,)(svundef_u8(), svundef_u8());
  // expected-error@+2 {{'svbgrp_n_u8' needs target feature sve,sve2,sve-bitperm}}
  // overload-error@+1 {{'svbgrp' needs target feature sve,sve2,sve-bitperm}}
  SVE_ACLE_FUNC(svbgrp,_n_u8,,)(svundef_u8(), u8);
  
  // expected-error@+2 {{'svbdep_u16' needs target feature sve,sve2,sve-bitperm}}
  // overload-error@+1 {{'svbdep' needs target feature sve,sve2,sve-bitperm}}
  SVE_ACLE_FUNC(svbdep,_u16,,)(svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svbdep_n_u16' needs target feature sve,sve2,sve-bitperm}}
  // overload-error@+1 {{'svbdep' needs target feature sve,sve2,sve-bitperm}}
  SVE_ACLE_FUNC(svbdep,_n_u16,,)(svundef_u16(), u16);
  // expected-error@+2 {{'svbext_u16' needs target feature sve,sve2,sve-bitperm}}
  // overload-error@+1 {{'svbext' needs target feature sve,sve2,sve-bitperm}}
  SVE_ACLE_FUNC(svbext,_u16,,)(svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svbext_n_u16' needs target feature sve,sve2,sve-bitperm}}
  // overload-error@+1 {{'svbext' needs target feature sve,sve2,sve-bitperm}}
  SVE_ACLE_FUNC(svbext,_n_u16,,)(svundef_u16(), u16);
  // expected-error@+2 {{'svbgrp_u16' needs target feature sve,sve2,sve-bitperm}}
  // overload-error@+1 {{'svbgrp' needs target feature sve,sve2,sve-bitperm}}
  SVE_ACLE_FUNC(svbgrp,_u16,,)(svundef_u16(), svundef_u16());
  // expected-error@+2 {{'svbgrp_n_u16' needs target feature sve,sve2,sve-bitperm}}
  // overload-error@+1 {{'svbgrp' needs target feature sve,sve2,sve-bitperm}}
  SVE_ACLE_FUNC(svbgrp,_n_u16,,)(svundef_u16(), u16);
  
  // expected-error@+2 {{'svbdep_u32' needs target feature sve,sve2,sve-bitperm}}
  // overload-error@+1 {{'svbdep' needs target feature sve,sve2,sve-bitperm}}
  SVE_ACLE_FUNC(svbdep,_u32,,)(svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svbdep_n_u32' needs target feature sve,sve2,sve-bitperm}}
  // overload-error@+1 {{'svbdep' needs target feature sve,sve2,sve-bitperm}}
  SVE_ACLE_FUNC(svbdep,_n_u32,,)(svundef_u32(), u32);
  // expected-error@+2 {{'svbext_u32' needs target feature sve,sve2,sve-bitperm}}
  // overload-error@+1 {{'svbext' needs target feature sve,sve2,sve-bitperm}}
  SVE_ACLE_FUNC(svbext,_u32,,)(svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svbext_n_u32' needs target feature sve,sve2,sve-bitperm}}
  // overload-error@+1 {{'svbext' needs target feature sve,sve2,sve-bitperm}}
  SVE_ACLE_FUNC(svbext,_n_u32,,)(svundef_u32(), u32);
  // expected-error@+2 {{'svbgrp_u32' needs target feature sve,sve2,sve-bitperm}}
  // overload-error@+1 {{'svbgrp' needs target feature sve,sve2,sve-bitperm}}
  SVE_ACLE_FUNC(svbgrp,_u32,,)(svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svbgrp_n_u32' needs target feature sve,sve2,sve-bitperm}}
  // overload-error@+1 {{'svbgrp' needs target feature sve,sve2,sve-bitperm}}
  SVE_ACLE_FUNC(svbgrp,_n_u32,,)(svundef_u32(), u32);
  // expected-error@+2 {{'svsm4e_u32' needs target feature sve,sve2-sm4}}
  // overload-error@+1 {{'svsm4e' needs target feature sve,sve2-sm4}}
  SVE_ACLE_FUNC(svsm4e,_u32,,)(svundef_u32(), svundef_u32());
  // expected-error@+2 {{'svsm4ekey_u32' needs target feature sve,sve2-sm4}}
  // overload-error@+1 {{'svsm4ekey' needs target feature sve,sve2-sm4}}
  SVE_ACLE_FUNC(svsm4ekey,_u32,,)(svundef_u32(), svundef_u32());
  
  // expected-error@+2 {{'svbdep_u64' needs target feature sve,sve2,sve-bitperm}}
  // overload-error@+1 {{'svbdep' needs target feature sve,sve2,sve-bitperm}}
  SVE_ACLE_FUNC(svbdep,_u64,,)(svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svbdep_n_u64' needs target feature sve,sve2,sve-bitperm}}
  // overload-error@+1 {{'svbdep' needs target feature sve,sve2,sve-bitperm}}
  SVE_ACLE_FUNC(svbdep,_n_u64,,)(svundef_u64(), u64);
  // expected-error@+2 {{'svbext_u64' needs target feature sve,sve2,sve-bitperm}}
  // overload-error@+1 {{'svbext' needs target feature sve,sve2,sve-bitperm}}
  SVE_ACLE_FUNC(svbext,_u64,,)(svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svbext_n_u64' needs target feature sve,sve2,sve-bitperm}}
  // overload-error@+1 {{'svbext' needs target feature sve,sve2,sve-bitperm}}
  SVE_ACLE_FUNC(svbext,_n_u64,,)(svundef_u64(), u64);
  // expected-error@+2 {{'svbgrp_u64' needs target feature sve,sve2,sve-bitperm}}
  // overload-error@+1 {{'svbgrp' needs target feature sve,sve2,sve-bitperm}}
  SVE_ACLE_FUNC(svbgrp,_u64,,)(svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svbgrp_n_u64' needs target feature sve,sve2,sve-bitperm}}
  // overload-error@+1 {{'svbgrp' needs target feature sve,sve2,sve-bitperm}}
  SVE_ACLE_FUNC(svbgrp,_n_u64,,)(svundef_u64(), u64);
  // expected-error@+2 {{'svpmullb_pair_u64' needs target feature sve,sve2,sve-aes}}
  // overload-error@+1 {{'svpmullb_pair' needs target feature sve,sve2,sve-aes}}
  SVE_ACLE_FUNC(svpmullb_pair,_u64,,)(svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svpmullb_pair_n_u64' needs target feature sve,sve2,sve-aes}}
  // overload-error@+1 {{'svpmullb_pair' needs target feature sve,sve2,sve-aes}}
  SVE_ACLE_FUNC(svpmullb_pair,_n_u64,,)(svundef_u64(), u64);
  // expected-error@+2 {{'svpmullt_pair_u64' needs target feature sve,sve2,sve-aes}}
  // overload-error@+1 {{'svpmullt_pair' needs target feature sve,sve2,sve-aes}}
  SVE_ACLE_FUNC(svpmullt_pair,_u64,,)(svundef_u64(), svundef_u64());
  // expected-error@+2 {{'svpmullt_pair_n_u64' needs target feature sve,sve2,sve-aes}}
  // overload-error@+1 {{'svpmullt_pair' needs target feature sve,sve2,sve-aes}}
  SVE_ACLE_FUNC(svpmullt_pair,_n_u64,,)(svundef_u64(), u64);
  // expected-error@+2 {{'svrax1_u64' needs target feature sve,sve2-sha3}}
  // overload-error@+1 {{'svrax1' needs target feature sve,sve2-sha3}}
  SVE_ACLE_FUNC(svrax1,_u64,,)(svundef_u64(), svundef_u64());

  // expected-error@+2 {{'svrax1_s64' needs target feature sve,sve2-sha3}}
  // overload-error@+1 {{'svrax1' needs target feature sve,sve2-sha3}}
  SVE_ACLE_FUNC(svrax1,_s64,,)(svundef_s64(), svundef_s64());
}
