// REQUIRES: mips-registered-target
// RUN: %clang_cc1 -triple mips-unknown-linux-gnu -fsyntax-only -verify %s

void foo() {
  // MIPS DSP Rev 1

  int a = 3;
  __builtin_mips_wrdsp(2052, a);  // expected-error{{argument to '__builtin_mips_wrdsp' must be a constant integer}}
  __builtin_mips_rddsp(a);        // expected-error{{argument to '__builtin_mips_rddsp' must be a constant integer}}
  __builtin_mips_wrdsp(2052, -1); // expected-error-re{{argument value {{.*}} is outside the valid range}}
  __builtin_mips_rddsp(-1);       // expected-error-re{{argument value {{.*}} is outside the valid range}}
  __builtin_mips_wrdsp(2052, 64); // expected-error-re{{argument value {{.*}} is outside the valid range}}
  __builtin_mips_rddsp(64);       // expected-error-re{{argument value {{.*}} is outside the valid range}}

  // MIPS DSP Rev 2

  __builtin_mips_append(1, 2, a); // expected-error{{argument to '__builtin_mips_append' must be a constant integer}}
  __builtin_mips_balign(1, 2, a); // expected-error{{argument to '__builtin_mips_balign' must be a constant integer}}
  __builtin_mips_precr_sra_ph_w(1, 2, a);   // expected-error{{argument to '__builtin_mips_precr_sra_ph_w' must be a constant integer}}
  __builtin_mips_precr_sra_r_ph_w(1, 2, a); // expected-error{{argument to '__builtin_mips_precr_sra_r_ph_w' must be a constant integer}}
  __builtin_mips_prepend(1, 2, a);          // expected-error{{argument to '__builtin_mips_prepend' must be a constant integer}}

  __builtin_mips_append(1, 2, -1);  // expected-error-re{{argument value {{.*}} is outside the valid range}}
  __builtin_mips_append(1, 2, 32);  // expected-error-re{{argument value {{.*}} is outside the valid range}}

  __builtin_mips_balign(1, 2, -1);  // expected-error-re{{argument value {{.*}} is outside the valid range}}
  __builtin_mips_balign(1, 2, 4);   // expected-error-re{{argument value {{.*}} is outside the valid range}}

  __builtin_mips_precr_sra_ph_w(1, 2, -1);  // expected-error-re{{argument value {{.*}} is outside the valid range}}
  __builtin_mips_precr_sra_ph_w(1, 2, 32);  // expected-error-re{{argument value {{.*}} is outside the valid range}}

  __builtin_mips_precr_sra_r_ph_w(1, 2, -1);  // expected-error-re{{argument value {{.*}} is outside the valid range}}
  __builtin_mips_precr_sra_r_ph_w(1, 2, 32);  // expected-error-re{{argument value {{.*}} is outside the valid range}}

  __builtin_mips_prepend(1, 2, -1); // expected-error-re{{argument value {{.*}} is outside the valid range}}
  __builtin_mips_prepend(1, 2, -1); // expected-error-re{{argument value {{.*}} is outside the valid range}}
}
