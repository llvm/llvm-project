// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple aarch64 -target-feature +sme -S -o /dev/null -Rpass-analysis=sme %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple aarch64 -target-feature +sme -S -o /dev/null -Rpass-analysis=sme %s -mllvm -aarch64-new-sme-abi 2>&1 | FileCheck %s --check-prefix=CHECK-NEWLOWERING

void private_za_callee_a();
void private_za_callee_b();
void private_za_callee_c();

void test_za_merge_paths(int a) __arm_inout("za") {
  if (a != 0)
    private_za_callee_a();
  else
    private_za_callee_b();
  // The new lowering won't report this call as the save is already needed due
  // to the call to `private_za_callee_*()` calls on both paths to this BB.
  private_za_callee_c();
}

void test_lazy_save_multiple_paths(int a) __arm_inout("za") {
  if (a != 0)
    private_za_callee_a();
  else {
    private_za_callee_b();
    // The new lowering won't report this call as the save is already needed due
    // to the call to `private_za_callee_b()`.
    private_za_callee_c();
  }
}

//      CHECK:  sme-remarks.c:14:5: remark: call from 'test_za_merge_paths' to 'unknown callee' sets up a lazy save for ZA [-Rpass-analysis=sme]
// CHECK-NEXT:     14 |     private_za_callee_b();
// CHECK-NEXT:        |     ^
// CHECK-NEXT:  sme-remarks.c:12:5: remark: call from 'test_za_merge_paths' to 'unknown callee' sets up a lazy save for ZA [-Rpass-analysis=sme]
// CHECK-NEXT:     12 |     private_za_callee_a();
// CHECK-NEXT:        |     ^
// CHECK-NEXT:  sme-remarks.c:17:3: remark: call from 'test_za_merge_paths' to 'unknown callee' sets up a lazy save for ZA [-Rpass-analysis=sme]
// CHECK-NEXT:     17 |   private_za_callee_c();
// CHECK-NEXT:        |   ^

//      CHECK:  sme-remarks.c:24:5: remark: call from 'test_lazy_save_multiple_paths' to 'unknown callee' sets up a lazy save for ZA [-Rpass-analysis=sme]
// CHECK-NEXT:     24 |     private_za_callee_b();
// CHECK-NEXT:        |     ^
// CHECK-NEXT:  sme-remarks.c:27:5: remark: call from 'test_lazy_save_multiple_paths' to 'unknown callee' sets up a lazy save for ZA [-Rpass-analysis=sme]
// CHECK-NEXT:     27 |     private_za_callee_c();
// CHECK-NEXT:        |     ^
// CHECK-NEXT:  sme-remarks.c:22:5: remark: call from 'test_lazy_save_multiple_paths' to 'unknown callee' sets up a lazy save for ZA [-Rpass-analysis=sme]
// CHECK-NEXT:     22 |     private_za_callee_a();
// CHECK-NEXT:        |     ^

//      CHECK-NEWLOWERING:  sme-remarks.c:11:9: remark: lazy save of ZA emitted in 'test_za_merge_paths' [-Rpass-analysis=sme]
// CHECK-NEWLOWERING-NEXT:    11 |   if (a != 0)
// CHECK-NEWLOWERING-NEXT:       |         ^
// CHECK-NEWLOWERING-NEXT:  sme-remarks.c:12:5: remark: call to 'private_za_callee_a' requires ZA save [-Rpass-analysis=sme]
// CHECK-NEWLOWERING-NEXT:    12 |     private_za_callee_a();
// CHECK-NEWLOWERING-NEXT:       |     ^
// CHECK-NEWLOWERING-NEXT:  sme-remarks.c:14:5: remark: call to 'private_za_callee_b' requires ZA save [-Rpass-analysis=sme]
// CHECK-NEWLOWERING-NEXT:    14 |     private_za_callee_b();
// CHECK-NEWLOWERING-NEXT:       |     ^

//      CHECK-NEWLOWERING:  sme-remarks.c:21:9: remark: lazy save of ZA emitted in 'test_lazy_save_multiple_paths' [-Rpass-analysis=sme]
// CHECK-NEWLOWERING-NEXT:    21 |   if (a != 0)
// CHECK-NEWLOWERING-NEXT:       |         ^
// CHECK-NEWLOWERING-NEXT:  sme-remarks.c:22:5: remark: call to 'private_za_callee_a' requires ZA save [-Rpass-analysis=sme]
// CHECK-NEWLOWERING-NEXT:    22 |     private_za_callee_a();
// CHECK-NEWLOWERING-NEXT:       |     ^
// CHECK-NEWLOWERING-NEXT:  sme-remarks.c:24:5: remark: call to 'private_za_callee_b' requires ZA save [-Rpass-analysis=sme]
// CHECK-NEWLOWERING-NEXT:    24 |     private_za_callee_b();
// CHECK-NEWLOWERING-NEXT:       |     ^
