// RUN: rm -rf %t
// RUN: %clang -fsyntax-only -isystem %S/Inputs/System/usr/include -fmodules -fmodule-cache-path %t -D__need_wint_t -Werror=implicit-function-declaration %s

// Supplied by compiler, but referenced from the "/usr/include" module map.
@__experimental_modules_import cstd.float_constants;

float getFltMax() { return FLT_MAX; }

// Supplied by the "/usr/include" module map.
@__experimental_modules_import cstd.stdio;

void test_fprintf(FILE *file) {
  fprintf(file, "Hello, modules\n");
}

// Supplied by compiler, which forwards to the "/usr/include" version.
@__experimental_modules_import cstd.stdint;

my_awesome_nonstandard_integer_type value;

// Supplied by the compiler; that version wins.
@__experimental_modules_import cstd.stdbool;

#ifndef bool
#  error "bool was not defined!"
#endif



