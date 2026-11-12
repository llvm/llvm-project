// RUN: %clang_cc1 -std=c++20 %s -emit-obj -o /dev/null

const int* test_odr_used() {
  // This previously crashed due to Value improperly being removed from
  // MaybeODRUseExprs.
  static constexpr int Value = 0;
  return __builtin_addressof(Value);
}
