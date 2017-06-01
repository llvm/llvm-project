// RUN: %clang_cc1 %s -ffake-address-space-map -verify -pedantic -fsyntax-only -DCONSTANT -cl-std=CL2.0
// RUN: %clang_cc1 %s -ffake-address-space-map -verify -pedantic -fsyntax-only -DGLOBAL -cl-std=CL2.0
// RUN: %clang_cc1 %s -ffake-address-space-map -verify -pedantic -fsyntax-only -DGENERIC -cl-std=CL2.0

/* OpenCLC v2.0 adds a set of restrictions for conversions between pointers to
*  different address spaces, mainly described in Sections 6.5.5 and 6.5.6.
*
*  It adds notion of overlapping address spaces. The main differention is that
*  an unnamed address space is added, called '__generic'. Pointers to the
*  generic address space can be interchangabley used with pointers to any
*  other address space except for __constant address space (Section 6.5.5).
*
*  Based on this there are 3 sets of tests: __generic, named (__global in this
*  case), and __constant, that should cover all program paths for CL address
*  space conversions used in initialisations, assignments, casts, comparisons
*  and arithmetic operations.
*/

#ifdef GENERIC
#define AS generic
#endif

#ifdef GLOBAL
#define AS global
#endif

#ifdef CONSTANT
#define AS constant
#endif

void f_glob(global int *arg_glob) {}
#ifndef GLOBAL
// expected-note@-2{{passing argument to parameter 'arg_glob' here}}
#endif

void f_loc(local int *arg_loc) {
} // expected-note@-1{{passing argument to parameter 'arg_loc' here}}

void f_const(constant int *arg_const) {}
#ifndef CONSTANT
// expected-note@-2{{passing argument to parameter 'arg_const' here}}
#endif

void f_priv(private int *arg_priv) {
} // expected-note@-1{{passing argument to parameter 'arg_priv' here}}

void f_gen(generic int *arg_gen) {}
#ifdef CONSTANT
// expected-note@-2{{passing argument to parameter 'arg_gen' here}}
#endif

void test_conversion(global int *arg_glob, local int *arg_loc,
                     constant int *arg_const, private int *arg_priv,
                     generic int *arg_gen) {

  AS int *var_init1 = arg_glob;
#ifdef CONSTANT
// expected-error@-2{{initializing '__constant int *' with an expression of type '__global int *' changes address space of pointer}}
#endif

  AS int *var_init2 = arg_loc;
#ifndef GENERIC
// expected-error-re@-2{{initializing '__{{global|constant}} int *' with an expression of type '__local int *' changes address space of pointer}}
#endif

  AS int *var_init3 = arg_const;
#ifndef CONSTANT
// expected-error-re@-2{{initializing '__{{global|generic}} int *' with an expression of type '__constant int *' changes address space of pointer}}
#endif

  AS int *var_init4 = arg_priv;
#ifndef GENERIC
// expected-error-re@-2{{initializing '__{{global|constant}} int *' with an expression of type 'int *' changes address space of pointer}}
#endif

  AS int *var_init5 = arg_gen;
#ifndef GENERIC
// expected-error-re@-2{{initializing '__{{global|constant}} int *' with an expression of type '__generic int *' changes address space of pointer}}
#endif

  AS int *var_cast1 = (AS int *)arg_glob;
#ifdef CONSTANT
// expected-error@-2{{casting '__global int *' to type '__constant int *' changes address space of pointer}}
#endif

  AS int *var_cast2 = (AS int *)arg_loc;
#ifndef GENERIC
// expected-error-re@-2{{casting '__local int *' to type '__{{global|constant}} int *' changes address space of pointer}}
#endif

  AS int *var_cast3 = (AS int *)arg_const;
#ifndef CONSTANT
// expected-error-re@-2{{casting '__constant int *' to type '__{{global|generic}} int *' changes address space of pointer}}
#endif

  AS int *var_cast4 = (AS int *)arg_priv;
#ifndef GENERIC
// expected-error-re@-2{{casting 'int *' to type '__{{global|constant}} int *' changes address space of pointer}}
#endif

  AS int *var_cast5 = (AS int *)arg_gen;
#ifdef CONSTANT
// expected-error@-2{{casting '__generic int *' to type '__constant int *' changes address space of pointer}}
#endif

  AS int *var_impl;
  var_impl = arg_glob;
#ifdef CONSTANT
// expected-error@-2{{assigning '__global int *' to '__constant int *' changes address space of pointer}}
#endif

  var_impl = arg_loc;
#ifndef GENERIC
// expected-error-re@-2{{assigning '__local int *' to '__{{global|constant}} int *' changes address space of pointer}}
#endif

  var_impl = arg_const;
#ifndef CONSTANT
// expected-error-re@-2{{assigning '__constant int *' to '__{{global|generic}} int *' changes address space of pointer}}
#endif

  var_impl = arg_priv;
#ifndef GENERIC
// expected-error-re@-2{{assigning 'int *' to '__{{global|constant}} int *' changes address space of pointer}}
#endif

  var_impl = arg_gen;
#ifndef GENERIC
// expected-error-re@-2{{assigning '__generic int *' to '__{{global|constant}} int *' changes address space of pointer}}
#endif

  var_cast1 = (AS int *)arg_glob;
#ifdef CONSTANT
// expected-error@-2{{casting '__global int *' to type '__constant int *' changes address space of pointer}}
#endif

  var_cast2 = (AS int *)arg_loc;
#ifndef GENERIC
// expected-error-re@-2{{casting '__local int *' to type '__{{global|constant}} int *' changes address space of pointer}}
#endif

  var_cast3 = (AS int *)arg_const;
#ifndef CONSTANT
// expected-error-re@-2{{casting '__constant int *' to type '__{{global|generic}} int *' changes address space of pointer}}
#endif

  var_cast4 = (AS int *)arg_priv;
#ifndef GENERIC
// expected-error-re@-2{{casting 'int *' to type '__{{global|constant}} int *' changes address space of pointer}}
#endif

  var_cast5 = (AS int *)arg_gen;
#ifdef CONSTANT
// expected-error@-2{{casting '__generic int *' to type '__constant int *' changes address space of pointer}}
#endif

  AS int *var_cmp;
  int b = var_cmp != arg_glob;
#ifdef CONSTANT
// expected-error@-2{{comparison between  ('__constant int *' and '__global int *') which are pointers to non-overlapping address spaces}}
#endif

  b = var_cmp != arg_loc;
#ifndef GENERIC
// expected-error-re@-2{{comparison between  ('__{{global|constant}} int *' and '__local int *') which are pointers to non-overlapping address spaces}}
#endif

  b = var_cmp == arg_const;
#ifndef CONSTANT
// expected-error-re@-2{{comparison between  ('__{{global|generic}} int *' and '__constant int *') which are pointers to non-overlapping address spaces}}
#endif

  b = var_cmp <= arg_priv;
#ifndef GENERIC
// expected-error-re@-2{{comparison between  ('__{{global|constant}} int *' and 'int *') which are pointers to non-overlapping address spaces}}
#endif

  b = var_cmp >= arg_gen;
#ifdef CONSTANT
// expected-error@-2{{comparison between  ('__constant int *' and '__generic int *') which are pointers to non-overlapping address spaces}}
#endif

  AS int *var_sub;
  b = var_sub - arg_glob;
#ifdef CONSTANT
// expected-error@-2{{arithmetic operation with operands of type  ('__constant int *' and '__global int *') which are pointers to non-overlapping address spaces}}
#endif

  b = var_sub - arg_loc;
#ifndef GENERIC
// expected-error-re@-2{{arithmetic operation with operands of type  ('__{{global|constant}} int *' and '__local int *') which are pointers to non-overlapping address spaces}}
#endif

  b = var_sub - arg_const;
#ifndef CONSTANT
// expected-error-re@-2{{arithmetic operation with operands of type  ('__{{global|generic}} int *' and '__constant int *') which are pointers to non-overlapping address spaces}}
#endif

  b = var_sub - arg_priv;
#ifndef GENERIC
// expected-error-re@-2{{arithmetic operation with operands of type  ('__{{global|constant}} int *' and 'int *') which are pointers to non-overlapping address spaces}}
#endif

  b = var_sub - arg_gen;
#ifdef CONSTANT
// expected-error@-2{{arithmetic operation with operands of type  ('__constant int *' and '__generic int *') which are pointers to non-overlapping address spaces}}
#endif

  f_glob(var_sub);
#ifndef GLOBAL
// expected-error-re@-2{{passing '__{{constant|generic}} int *' to parameter of type '__global int *' changes address space of pointer}}
#endif

  f_loc(var_sub); // expected-error-re{{passing '__{{global|constant|generic}} int *' to parameter of type '__local int *' changes address space of pointer}}

  f_const(var_sub);
#ifndef CONSTANT
// expected-error-re@-2{{passing '__{{global|generic}} int *' to parameter of type '__constant int *' changes address space of pointer}}
#endif

  f_priv(var_sub); // expected-error-re{{passing '__{{global|constant|generic}} int *' to parameter of type 'int *' changes address space of pointer}}

  f_gen(var_sub);
#ifdef CONSTANT
// expected-error@-2{{passing '__constant int *' to parameter of type '__generic int *' changes address space of pointer}}
#endif
}

void test_ternary() {
  AS int *var_cond;
  generic int *var_gen;
  global int *var_glob;
  var_gen = 0 ? var_cond : var_glob;
#ifdef CONSTANT
// expected-error@-2{{conditional operator with the second and third operands of type  ('__constant int *' and '__global int *') which are pointers to non-overlapping address spaces}}
#endif

  local int *var_loc;
  var_gen = 0 ? var_cond : var_loc;
#ifndef GENERIC
// expected-error-re@-2{{conditional operator with the second and third operands of type  ('__{{global|constant}} int *' and '__local int *') which are pointers to non-overlapping address spaces}}
#endif

  constant int *var_const;
  var_cond = 0 ? var_cond : var_const;
#ifndef CONSTANT
// expected-error-re@-2{{conditional operator with the second and third operands of type  ('__{{global|generic}} int *' and '__constant int *') which are pointers to non-overlapping address spaces}}
#endif

  private int *var_priv;
  var_gen = 0 ? var_cond : var_priv;
#ifndef GENERIC
// expected-error-re@-2{{conditional operator with the second and third operands of type  ('__{{global|constant}} int *' and 'int *') which are pointers to non-overlapping address spaces}}
#endif

  var_gen = 0 ? var_cond : var_gen;
#ifdef CONSTANT
// expected-error@-2{{conditional operator with the second and third operands of type  ('__constant int *' and '__generic int *') which are pointers to non-overlapping address spaces}}
#endif

  void *var_void_gen;
  global char *var_glob_ch;
  var_void_gen = 0 ? var_cond : var_glob_ch;
#ifdef CONSTANT
// expected-error@-2{{conditional operator with the second and third operands of type  ('__constant int *' and '__global char *') which are pointers to non-overlapping address spaces}}
#endif

  local char *var_loc_ch;
  var_void_gen = 0 ? var_cond : var_loc_ch;
#ifndef GENERIC
// expected-error-re@-2{{conditional operator with the second and third operands of type  ('__{{global|constant}} int *' and '__local char *') which are pointers to non-overlapping address spaces}}
#endif

  constant void *var_void_const;
  constant char *var_const_ch;
  var_void_const = 0 ? var_cond : var_const_ch;
#ifndef CONSTANT
// expected-error-re@-2{{conditional operator with the second and third operands of type  ('__{{global|generic}} int *' and '__constant char *') which are pointers to non-overlapping address spaces}}
#endif

  private char *var_priv_ch;
  var_void_gen = 0 ? var_cond : var_priv_ch;
#ifndef GENERIC
// expected-error-re@-2{{conditional operator with the second and third operands of type  ('__{{global|constant}} int *' and 'char *') which are pointers to non-overlapping address spaces}}
#endif

  generic char *var_gen_ch;
  var_void_gen = 0 ? var_cond : var_gen_ch;
#ifdef CONSTANT
// expected-error@-2{{conditional operator with the second and third operands of type  ('__constant int *' and '__generic char *') which are pointers to non-overlapping address spaces}}
#endif
}

