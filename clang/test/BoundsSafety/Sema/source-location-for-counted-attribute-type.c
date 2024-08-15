
// RUN: not %clang_cc1 -fbounds-safety -fsyntax-only %s 2>&1 | FileCheck %s
#include <ptrcheck.h>

// This test is essentially a test of
// `SourceLocationFor(const CountAttributedType *CATy, Sema &S)`. The test is
// checking the column number attached the diagnostic.
// FIXME: This should probably be a unit test. We can't test `__sized_by` or
// `__sized_by_or_null` here because the diagnostic we are relying isn't emitted
// for those attributes.

struct IncompleteTy; // Incomplete

//==============================================================================
// counted_by
//==============================================================================

#define custom_counted_by(X) __attribute__((counted_by(X)))

#define custom_counted_by__(X) __attribute__((__counted_by__(X)))

#define COUNT_ARG 0
struct CB {
    int count;
    // CHECK: [[@LINE+1]]:26: note: consider using '__sized_by' instead of '__counted_by'
    struct IncompleteTy* __counted_by(count) b_macro;
    // CHECK: [[@LINE+1]]:41: note: consider using '__sized_by' instead of '__counted_by'
    struct IncompleteTy* __attribute__((counted_by(count))) b_direct_non_affixed;
    // CHECK: [[@LINE+1]]:41: note: consider using '__sized_by' instead of '__counted_by'
    struct IncompleteTy* __attribute__((__counted_by__(count))) b_direct_affixed;
    // NOTE: In these cases the locations just point to the count expressions as
    // a fallback.
    // CHECK: [[@LINE+1]]:44: note: consider using '__sized_by' instead of '__counted_by'
    struct IncompleteTy* custom_counted_by(count) b_custom_macro;
    // CHECK: [[@LINE+1]]:46: note: consider using '__sized_by' instead of '__counted_by'
    struct IncompleteTy* custom_counted_by__(count) b_custom_macro_underscored;

    // CHECK: [[@LINE+1]]:39: note: consider using '__sized_by' instead of '__counted_by'
    struct IncompleteTy* __counted_by(COUNT_ARG) b_macro_macro_arg;
    // CHECK: [[@LINE+1]]:52: note: consider using '__sized_by' instead of '__counted_by'
    struct IncompleteTy* __attribute__((counted_by(COUNT_ARG))) b_direct_non_affixed_macro_arg;
    // CHECK: [[@LINE+1]]:56: note: consider using '__sized_by' instead of '__counted_by'
    struct IncompleteTy* __attribute__((__counted_by__(COUNT_ARG))) b_direct_affixed_macro_arg;
};

void useCB(struct CB* cb) {
    struct IncompleteTy* local0 = cb->b_macro;
    struct IncompleteTy* local1 = cb->b_direct_non_affixed;
    struct IncompleteTy* local2 = cb->b_direct_affixed;
    struct IncompleteTy* local3 = cb->b_custom_macro;
    struct IncompleteTy* local4 = cb->b_custom_macro_underscored;
    struct IncompleteTy* local5 = cb->b_macro_macro_arg;
    struct IncompleteTy* local6 = cb->b_direct_non_affixed_macro_arg;
    struct IncompleteTy* local7 = cb->b_direct_affixed_macro_arg;
}

//==============================================================================
// counted_by_or_null
//==============================================================================

#define custom_counted_by_or_null(X) __attribute__((counted_by_or_null(X)))
#define custom_counted_by_or_null__(X) __attribute__((__counted_by_or_null__(X)))

struct CBON {
    int count;
    // CHECK: [[@LINE+1]]:26: note: consider using '__sized_by_or_null' instead of '__counted_by_or_null'
    struct IncompleteTy* __counted_by_or_null(count) b_macro;
    // CHECK: [[@LINE+1]]:41: note: consider using '__sized_by_or_null' instead of '__counted_by_or_null'
    struct IncompleteTy* __attribute__((counted_by_or_null(count))) b_direct_non_affixed;
    // CHECK: [[@LINE+1]]:41: note: consider using '__sized_by_or_null' instead of '__counted_by_or_null'
    struct IncompleteTy* __attribute__((__counted_by_or_null__(count))) b_direct_affixed;
    // NOTE: In these cases the locations just points to the count expressions as
    // a fallback.
    // CHECK: [[@LINE+1]]:52: note: consider using '__sized_by_or_null' instead of '__counted_by_or_null'
    struct IncompleteTy* custom_counted_by_or_null(count) b_custom_macro;
    // CHECK: [[@LINE+1]]:54: note: consider using '__sized_by_or_null' instead of '__counted_by_or_null'
    struct IncompleteTy* custom_counted_by_or_null__(count) b_custom_macro_underscored;

    // CHECK: [[@LINE+1]]:47: note: consider using '__sized_by_or_null' instead of '__counted_by_or_null'
    struct IncompleteTy* __counted_by_or_null(COUNT_ARG) b_macro_macro_arg;
    // CHECK: [[@LINE+1]]:60: note: consider using '__sized_by_or_null' instead of '__counted_by_or_null'
    struct IncompleteTy* __attribute__((counted_by_or_null(COUNT_ARG))) b_direct_non_affixed_macro_arg;
    // CHECK: [[@LINE+1]]:64: note: consider using '__sized_by_or_null' instead of '__counted_by_or_null'
    struct IncompleteTy* __attribute__((__counted_by_or_null__(COUNT_ARG))) b_direct_affixed_macro_arg;
};


void useCBON(struct CBON* cbon) {
    struct IncompleteTy* local0 = cbon->b_macro;
    struct IncompleteTy* local1 = cbon->b_direct_non_affixed;
    struct IncompleteTy* local2 = cbon->b_direct_affixed;
    struct IncompleteTy* local3 = cbon->b_custom_macro;
    struct IncompleteTy* local4 = cbon->b_custom_macro_underscored;
    struct IncompleteTy* local5 = cbon->b_macro_macro_arg;
    struct IncompleteTy* local6 = cbon->b_direct_non_affixed_macro_arg;
    struct IncompleteTy* local7 = cbon->b_direct_affixed_macro_arg;
}
