// RUN: %clang_cc1 %s -fopenacc -verify -Wno-vla-cxx-extension

// C++ companion to compute-construct-reduction-vla.c.
//
// Regression test for llvm/llvm-project#199162.  The original reproducer in
// that issue was driven through clang++, so we keep an explicit C++
// counterpart that exercises the same Sema path on:
//   * a bare VLA declared in a function body (clang accepts as a GNU
//     extension in C++; gated by -Wno-vla-cxx-extension above),
//   * references to VLAs,
//   * VLAs reached through a function template (dependent then concrete
//     after instantiation),
//   * an array section over a VLA.
//
// All cases must parse cleanly without crashing in
// GenerateReductionInitRecipeExpr.

// expected-no-diagnostics

void vla_reduction_bitand_cxx(int i) {
  int arr[i + 1];
#pragma acc parallel reduction(& : arr)
  while (1)
    ;
}

void vla_reduction_add_cxx(int i) {
  int arr[i + 1];
#pragma acc parallel reduction(+ : arr)
  while (1)
    ;
}

void vla_reduction_serial_cxx(int i) {
  unsigned arr[i + 1];
#pragma acc serial reduction(| : arr)
  while (1)
    ;
}

// A reference binding to a VLA — VarTy after getNonReferenceType() is
// still a VariableArrayType, so the recipe builder runs the same path.
void vla_reduction_reference(int i) {
  int storage[i + 1];
  int(&arr)[i + 1] = storage;
#pragma acc parallel reduction(+ : arr)
  while (1)
    ;
}

// Template — the VLA size depends on a non-type template parameter, so
// the operand type is dependent at template-definition time and
// CreateReductionInitRecipe should bail out early via the
// isDependentType() guard.  After instantiation the type becomes
// concrete (still a VLA: the runtime size depends on 'n' below), and
// the recipe builder runs the punt branch we added.
template <int Pad>
void vla_reduction_template(int n) {
  int arr[n + Pad];
#pragma acc parallel reduction(+ : arr)
  while (1)
    ;
}

void instantiate_template(int n) {
  vla_reduction_template<1>(n);
  vla_reduction_template<8>(n);
}

// 2D VLA, in C++.
void vla_reduction_2d_cxx(int i, int j) {
  int arr[i + 1][j + 1];
#pragma acc parallel reduction(+ : arr)
  while (1)
    ;
}

// Array section on a VLA, in C++.
void vla_reduction_array_section_cxx(int i) {
  int arr[i + 1];
#pragma acc parallel reduction(+ : arr[0:i])
  while (1)
    ;
}

// Combined construct.
void vla_reduction_combined_cxx(int i) {
  int arr[i + 1];
#pragma acc parallel loop reduction(+ : arr)
  for (int k = 0; k < 10; ++k)
    ;
}
