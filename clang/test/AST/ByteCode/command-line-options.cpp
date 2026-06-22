/// This tests that the bytecode interpreter is in use if -fexperimental-new-constant-interpreter is passed.
/// This should be the case regardless of whether CLANG_USE_EXPERIMENTAL_CONST_INTERP is enabled or not.
///
/// Similarly, it should _not_ be used if -fno-experimental-new-constant-interpreter is passed.
///
/// All this should be true if the driver is used or -cc1.


// RUN: %clang -c   -fexperimental-new-constant-interpreter %s -Xclang -verify=bc
// RUN: %clang -cc1 -fexperimental-new-constant-interpreter %s         -verify=bc

// RUN: %clang -c   -fno-experimental-new-constant-interpreter %s -Xclang -verify=nobc
// RUN: %clang -cc1 -fno-experimental-new-constant-interpreter %s         -verify=nobc


/// Note that we're not testing the behavior without those command line options since that
/// depends on the value of CLANG_USE_EXPERIMENTAL_CONST_INTERP, which we can't test for.


// bc-no-diagnostics


/// We test for the bytecode interperter by trying to bitcast a bitfield.
struct S {
  unsigned a : 10;
};
constexpr S s = __builtin_bit_cast(S, 12); // nobc-error {{must be initialized by a constant expression}} \
                                           // nobc-note {{constexpr bit_cast involving bit-field is not yet supported}}
