// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple aarch64 -target-feature +sme -mllvm -aarch64-new-sme-abi=false -Rpass-analysis=sme -verify=expected-sdag %s -S -o /dev/null
// RUN: %clang_cc1 -triple aarch64 -target-feature +sme -Rpass-analysis=sme -verify %s -S -o /dev/null %s

void private_za_callee();

void test_save_remarks(int a) __arm_inout("za") {
  // expected-sdag-remark@+3 {{call from 'test_save_remarks' to 'unknown callee' sets up a lazy save for ZA}}
  // expected-remark@+2 {{lazy save of ZA emitted in 'test_save_remarks'}}
  // expected-remark@+1 {{call to 'private_za_callee' requires ZA save}}
  private_za_callee();
}
