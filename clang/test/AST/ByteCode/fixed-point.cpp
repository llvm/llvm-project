// RUN: %clang_cc1 %s -fsyntax-only -ffixed-point -verify=expected,both -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 %s -fsyntax-only -ffixed-point -verify=ref,both

static_assert((bool)1.0k);
static_assert(!((bool)0.0k));
static_assert((bool)0.0k); // both-error {{static assertion failed}}

static_assert(1.0k == 1.0k);
static_assert(1.0k != 1.0k); // both-error {{failed due to requirement '1.0k != 1.0k'}}
static_assert(-12.0k == -(-(-12.0k)));

