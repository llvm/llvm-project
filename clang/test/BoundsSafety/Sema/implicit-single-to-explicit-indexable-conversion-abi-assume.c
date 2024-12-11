
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s

// This is a test for
// "-Wbounds-attributes-implicit-conversion-single-to-explicit-indexable"'s
// behavior when the `__ptrcheck_abi_assume_*()` macros are used.
//
// rdar://91980829
// Currently this warning is not emitted for the code in this test. Ideally we
// would emit warnings here but the current implementation looks for
// `attr::PtrAutoAttr` to make sure we don't warn on assignment to local
// variables that are implicitly `__bidi_indexable`. Unfortunately we see
// exactly the same attribute for ABI visible pointers that are implicitly
// annotated using __ptrcheck_abi_assume_bidi_indexable() or
// __ptrcheck_abi_assume_indexable(). Thus we cannot distinguish them and so no
// warnings are emitted.

// expected-no-diagnostics

#include <ptrcheck.h>
extern int *__single explicitly_single0;

//--------------------------------------------------------------------------
// Assume __bidi_indexable on ABI pointers
//--------------------------------------------------------------------------
__ptrcheck_abi_assume_bidi_indexable();

void take_bidi0(int *b_arg0);
int *implicitly_bidi0;

typedef struct {
  int *field;
} BidiStruct_t;

void use_bidi(void) {
  // Initialization
  BidiStruct_t bidi = {.field = explicitly_single0};
  // Assignment
  implicitly_bidi0 = explicitly_single0;
  bidi.field = explicitly_single0;
  // Argument pass
  take_bidi0(explicitly_single0);
}

int *no_warn_ret_bidi(void) {
  return explicitly_single0;
}

//--------------------------------------------------------------------------
// Assume __indexable on ABI pointers
//--------------------------------------------------------------------------
__ptrcheck_abi_assume_indexable();

void take_idx0(int *i_arg0);
int *implicitly_idx0;

typedef struct {
  int *field;
} IdxStruct_t;

void use_idx(void) {
  // Initialization
  IdxStruct_t idx = {.field = explicitly_single0};
  // Assignment
  implicitly_idx0 = explicitly_single0;
  idx.field = explicitly_single0;
  // Argument pass
  take_idx0(explicitly_single0);
}

int *no_warn_ret_idx(void) {
  return explicitly_single0;
}
