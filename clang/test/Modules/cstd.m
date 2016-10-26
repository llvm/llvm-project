// RUN: rm -rf %t
// RUN: %clang_cc1 -fsyntax-only -internal-isystem %S/Inputs/System/usr/include -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -D__need_wint_t -Werror=implicit-function-declaration %s

@import uses_other_constants;
const double other_value = DBL_MAX;

// Supplied by compiler, but referenced from the "/usr/include" module map.
@import cstd.float_constants;

float getFltMax() { return FLT_MAX; }

// Supplied by the "/usr/include" module map.
@import cstd.stdio;

void test_fprintf(FILE *file) {
  fprintf(file, "Hello, modules\n");
}

// Supplied by compiler, which forwards to the "/usr/include" version.
@import cstd.stdint;

my_awesome_nonstandard_integer_type value2;

// Supplied by the compiler; that version wins.
@import cstd.stdbool;

#ifndef bool
#  error "bool was not defined!"
#endif

