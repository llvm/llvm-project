// RUN: %clang_cc1 %s -fopenacc -verify

// Regression test for llvm/llvm-project#199162:
//
// A variable-length array operand to an OpenACC 'reduction' clause used to
// trip the assertion `Ty->isScalarType()` inside
// GenerateReductionInitRecipeExpr (clang/lib/Sema/SemaOpenACC.cpp).
//
// VLAs cannot be statically enumerated to build an InitListExpr, so we now
// punt (the same way we already do for pointer types) and let codegen handle
// it.  This test makes sure none of the variations below crash in Sema and
// that they parse cleanly without spurious diagnostics.

// expected-no-diagnostics

void vla_reduction_bitand(int i) {
  int arr[i + 1];
#pragma acc parallel reduction(& : arr)
  while (1)
    ;
}

void vla_reduction_add(int i) {
  int arr[i + 1];
#pragma acc parallel reduction(+ : arr)
  while (1)
    ;
}

void vla_reduction_max(int i) {
  int arr[i + 1];
#pragma acc parallel reduction(max : arr)
  while (1)
    ;
}

void vla_reduction_min(int i) {
  int arr[i + 1];
#pragma acc parallel reduction(min : arr)
  while (1)
    ;
}

void vla_reduction_mul(int i) {
  int arr[i + 1];
#pragma acc parallel reduction(* : arr)
  while (1)
    ;
}

void vla_reduction_bitor(int i) {
  unsigned arr[i + 1];
#pragma acc parallel reduction(| : arr)
  while (1)
    ;
}

void vla_reduction_bitxor(int i) {
  unsigned arr[i + 1];
#pragma acc parallel reduction(^ : arr)
  while (1)
    ;
}

void vla_reduction_serial(int i) {
  int arr[i + 1];
#pragma acc serial reduction(| : arr)
  while (1)
    ;
}
