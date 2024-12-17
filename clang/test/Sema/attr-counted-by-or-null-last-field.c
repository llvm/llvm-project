// RUN: %clang_cc1 -fsyntax-only -verify=expected,immediate %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-late-parse-attributes -verify=expected,late %s

#define __counted_by_or_null(f)  __attribute__((counted_by_or_null(f)))

// This has been adapted from clang/test/Sema/attr-counted-by-vla.c, but with VLAs replaced with pointers

struct bar;

struct not_found {
  int count;
  struct bar *ptr __counted_by_or_null(bork); // expected-error {{use of undeclared identifier 'bork'}}
};

struct no_found_count_not_in_substruct {
  unsigned long flags;
  unsigned char count; // expected-note {{'count' declared here}}
  struct A {
    int dummy;
    int * ptr __counted_by_or_null(count); // expected-error {{'counted_by_or_null' field 'count' isn't within the same struct as the annotated pointer}}
  } a;
};

struct not_found_count_not_in_unnamed_substruct {
  unsigned char count; // expected-note {{'count' declared here}}
  struct {
    int dummy;
    int * ptr __counted_by_or_null(count); // expected-error {{'counted_by_or_null' field 'count' isn't within the same struct as the annotated pointer}}
  } a;
};

struct not_found_count_not_in_unnamed_substruct_2 {
  struct {
    unsigned char count; // expected-note {{'count' declared here}}
  };
  struct {
    int dummy;
    int * ptr __counted_by_or_null(count); // expected-error {{'counted_by_or_null' field 'count' isn't within the same struct as the annotated pointer}}
  } a;
};

struct not_found_count_in_other_unnamed_substruct {
  struct {
    unsigned char count;
  } a1;

  struct {
    int dummy;
    int * ptr __counted_by_or_null(count); // expected-error {{use of undeclared identifier 'count'}}
  };
};

struct not_found_count_in_other_substruct {
  struct _a1 {
    unsigned char count;
  } a1;

  struct {
    int dummy;
    int * ptr __counted_by_or_null(count); // expected-error {{use of undeclared identifier 'count'}}
  };
};

struct not_found_count_in_other_substruct_2 {
  struct _a2 {
    unsigned char count;
  } a2;

  int * ptr __counted_by_or_null(count); // expected-error {{use of undeclared identifier 'count'}}
};

struct not_found_suggest {
  int bork;
  struct bar **ptr __counted_by_or_null(blork); // expected-error {{use of undeclared identifier 'blork'}}
};

int global; // expected-note {{'global' declared here}}

struct found_outside_of_struct {
  int bork;
  struct bar ** ptr __counted_by_or_null(global); // expected-error {{field 'global' in 'counted_by_or_null' not inside structure}}
};

struct self_referrential {
  int bork;
  // immediate-error@+2{{use of undeclared identifier 'self'}}
  // late-error@+1{{'counted_by_or_null' only applies to pointers; did you mean to use 'counted_by'?}}
  struct bar *self[] __counted_by_or_null(self);
};

struct non_int_count {
  double dbl_count;
  struct bar ** ptr __counted_by_or_null(dbl_count); // expected-error {{'counted_by_or_null' requires a non-boolean integer type argument}}
};

struct array_of_ints_count {
  int integers[2];
  struct bar ** ptr __counted_by_or_null(integers); // expected-error {{'counted_by_or_null' requires a non-boolean integer type argument}}
};

struct not_a_c99_fam {
  int count;
  struct bar *non_c99_fam[0] __counted_by_or_null(count); // expected-error {{'counted_by_or_null' only applies to pointers; did you mean to use 'counted_by'?}}
};

struct annotated_with_anon_struct {
  unsigned long flags;
  struct {
    unsigned char count;
    int * ptr __counted_by_or_null(crount); // expected-error {{use of undeclared identifier 'crount'}}
  };
};

//==============================================================================
// __counted_by_or_null on a struct ptr with element type that has unknown count
//==============================================================================

struct count_unknown;
struct on_member_ptr_incomplete_ty_ty_pos {
  int count;
  struct count_unknown * ptr __counted_by_or_null(count); // expected-error {{'counted_by_or_null' cannot be applied to a pointer with pointee of unknown size because 'struct count_unknown' is an incomplete type}}
};

struct on_member_ptr_incomplete_const_ty_ty_pos {
  int count;
  const struct count_unknown * ptr __counted_by_or_null(count); // expected-error {{'counted_by_or_null' cannot be applied to a pointer with pointee of unknown size because 'const struct count_unknown' is an incomplete type}}
};

struct on_member_ptr_void_ty_ty_pos {
  int count;
  void * ptr __counted_by_or_null(count); // expected-error {{'counted_by_or_null' cannot be applied to a pointer with pointee of unknown size because 'void' is an incomplete type}}
};

typedef void(fn_ty)(int);

struct on_member_ptr_fn_ptr_ty {
  int count;
  fn_ty* * ptr __counted_by_or_null(count);
};

struct on_member_ptr_fn_ty {
  int count;
  fn_ty * ptr __counted_by_or_null(count); // expected-error {{'counted_by_or_null' cannot be applied to a pointer with pointee of unknown size because 'fn_ty' (aka 'void (int)') is a function type}}
};
