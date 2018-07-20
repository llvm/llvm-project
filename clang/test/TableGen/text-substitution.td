// RUN: clang-tblgen -gen-clang-diags-defs -I%S %s -o - 2>&1 | \
// RUN:    FileCheck --strict-whitespace %s
include "DiagnosticBase.inc"

def yes_no : TextSubstitution<"%select{yes|no}0">;
def says_yes : TextSubstitution<"%1 says %sub{yes_no}0">;


def sub_test_rewrite : TextSubstitution<
  "SELECT! %select{one|two}3. "
  "DIFF! %diff{$ is $|or not}0,1. "
  "PLURAL! %plural{0:zero items|[1,2]:one or two item|:multiple items}2. "
  "ORDINAL! %ordinal1. "
  "S! item%s2. "
  "Q! %q4. "
  "PLACEHOLDER! %5."
  "OBJCCLASS! %objcclass0. "
  "OBJCINSTANCE! %objcinstance1. ">;

// CHECK: DIAG(test_rewrite,
// CHECK-SAME: SELECT! %select{one|two}2.
// CHECK-SAME: DIFF! %diff{$ is $|or not}5,4.
// CHECK-SAME: PLURAL! %plural{0:zero items|[1,2]:one or two item|:multiple items}3.
// CHECK-SAME: ORDINAL! %ordinal4.
// CHECK-SAME: S! item%s3.
// CHECK-SAME: Q! %q1.
// CHECK-SAME: PLACEHOLDER! %0.OBJCCLASS!
// CHECK-SAME: %objcclass5. OBJCINSTANCE!
// CHECK-SAME: %objcinstance4.  DONE!",
def test_rewrite: Error<"%sub{sub_test_rewrite}5,4,3,2,1,0 DONE!">;

def test_sub_basic : Error<"%sub{yes_no}0">;
// CHECK: test_sub_basic
// CHECK-SAME: "%select{yes|no}0",

def test_sub_nested : Error<"%sub{says_yes}2,4">;
// CHECK: test_sub_nested
// CHECK-SAME: "%4 says %select{yes|no}2",
