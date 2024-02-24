// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify=both,expected %s
// RUN: %clang_cc1 -verify=both,ref %s

/// FIXME: Copied from test/Sema/atomic-expr.c.
/// this expression seems to be rejected for weird reasons,
/// but we imitate the current interpreter's behavior.
_Atomic int ai = 0;
// FIXME: &ai is an address constant, so this should be accepted as an
// initializer, but the bit-cast inserted due to the pointer conversion is
// tripping up the test for whether the initializer is a constant expression.
// The warning is correct but the error is not.
_Atomic(int *) aip3 = &ai; // both-warning {{incompatible pointer types initializing '_Atomic(int *)' with an expression of type '_Atomic(int) *'}} \
                           // both-error {{initializer element is not a compile-time constant}}
