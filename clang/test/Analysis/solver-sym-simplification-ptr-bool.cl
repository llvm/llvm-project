// RUN: %clang_analyze_cc1 -triple amdgcn-unknown-unknown -analyzer-checker=core %s

// expected-no-diagnostics

// This test case covers an issue found in the static analyzer
// solver where pointer sizes were assumed. Pointer sizes may vary on other
// architectures. This issue was originally discovered on a downstream,
// custom target, this assert occurs on the custom target and this one
// without the fix, and is fixed with this change.
//
// The assertion appears to be happening as a result of evaluating the
// SymIntExpr (reg_$0<int * p>) != 0U in VisitSymIntExpr located in
// SimpleSValBuilder.cpp. The LHS is evaluated to 32b and the RHS is
// evaluated to 16b. This eventually leads to the assertion in APInt.h.
//
// APInt.h:1151: bool llvm::APInt::operator==(const llvm::APInt &) const: Assertion `BitWidth == RHS.BitWidth && "Comparison requires equal bit widths"'
// 
void test1(__attribute__((address_space(256))) int * p) {
  __attribute__((address_space(256))) int * q = p-1;
  if (q) {}
  if (q) {}
  (void)q;
}
 
void test2(__attribute__((address_space(256))) int * p) {
  __attribute__((address_space(256))) int * q = p-1;
  q && q; 
  q && q; 
  (void)q;
} 
