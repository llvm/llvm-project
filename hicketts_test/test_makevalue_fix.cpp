// Test cases for the fix in llvm/llvm-project#144313
//
// This exercises the bugprone-unchecked-optional-access check's handling
// of BloombergLP::bdlb::NullableValue::makeValue and makeValueInplace.
//
// Before the fix, cases marked "OK" below would produce false-positive
// warnings because the checker didn't know makeValue/makeValueInplace
// establish a valid value.
//
// Run with:
//   clang-tidy -checks='bugprone-unchecked-optional-access' \
//     test_makevalue_fix.cpp -- \
//     -I <llvm-project>/clang-tools-extra/test/clang-tidy/checkers/bugprone/Inputs/unchecked-optional-access

#include "bde/types/bdlb_nullablevalue.h"

// -- SHOULD NOT WARN (the fix) --

// makeValue(val) on null branch guarantees value is present on all paths.
void makeValue_covers_null_branch(BloombergLP::bdlb::NullableValue<int> &opt) {
  if (opt.isNull()) {
    opt.makeValue(42);
  }
  opt.value(); // OK — either was non-null already, or makeValue filled it
}

// makeValueInplace does the same thing via in-place construction.
void makeValueInplace_covers_null_branch(BloombergLP::bdlb::NullableValue<int> &opt) {
  if (opt.isNull()) {
    opt.makeValueInplace(42);
  }
  opt.value(); // OK
}

// Unconditional makeValue — always safe to access afterwards.
void unconditional_makeValue(BloombergLP::bdlb::NullableValue<int> &opt) {
  opt.makeValue(100);
  opt.value(); // OK
}

// Zero-arg makeValue — default-constructs the value.
void makeValue_no_args(BloombergLP::bdlb::NullableValue<int> &opt) {
  opt.makeValue();
  opt.value(); // OK
}

// -- SHOULD WARN (not fixed by makeValue) --

// Accessing without any check or makeValue is still unsafe.
void no_check_no_makeValue(BloombergLP::bdlb::NullableValue<int> &opt) {
  opt.value(); // WARNING: unchecked access to optional value
}

// reset() after makeValue invalidates the value.
void makeValue_then_reset(BloombergLP::bdlb::NullableValue<int> &opt) {
  opt.makeValue(42);
  opt.reset();
  opt.value(); // WARNING: value was reset
}

// makeValue on a *different* object doesn't help.
void makeValue_wrong_object(BloombergLP::bdlb::NullableValue<int> &a,
                            BloombergLP::bdlb::NullableValue<int> &b) {
  if (a.isNull()) {
    b.makeValue(42);
  }
  a.value(); // WARNING: a may still be null
}
