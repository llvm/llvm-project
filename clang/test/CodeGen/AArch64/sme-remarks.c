// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64 -target-feature +sme -Rpass-analysis=sme -verify %s -S -o /dev/null %s

void private_za_callee_a();
void private_za_callee_b();
void private_za_callee_c();

void test_za_merge_paths(int a) __arm_inout("za") {
  // expected-remark@+1 {{lazy save of ZA emitted in 'test_za_merge_paths'}}
  if (a != 0)
    // expected-remark@+1 {{call to 'private_za_callee_a' requires ZA save}}
    private_za_callee_a();
  else
    // expected-remark@+1 {{call to 'private_za_callee_b' requires ZA save}}
    private_za_callee_b();
  /// The analysis won't report this call as the save is already needed due to
  /// the call to `private_za_callee_a/b()` calls on both paths to this call.
  private_za_callee_c();
}

void test_lazy_save_multiple_paths(int a) __arm_inout("za") {
  // expected-remark@+1 {{lazy save of ZA emitted in 'test_lazy_save_multiple_paths'}}
  if (a != 0)
    // expected-remark@+1 {{call to 'private_za_callee_a' requires ZA save}}
    private_za_callee_a();
  else {
    // expected-remark@+1 {{call to 'private_za_callee_b' requires ZA save}}
    private_za_callee_b();
    /// The analysis won't report this call as the save is already needed due
    /// to the call to `private_za_callee_b()`.
    private_za_callee_c();
  }
}
