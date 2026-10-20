// RUN: %clang_cc1 -fsyntax-only -verify=expected,immediate %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-late-parse-attributes -verify=expected,late %s

#define __sized_by_or_null(f)  __attribute__((sized_by_or_null(f)))

// This has been adapted from clang/test/Sema/attr-counted-by-vla.c, but with VLAs replaced with pointers

struct bar;

struct not_found {
  int size;
  struct bar *ptr __sized_by_or_null(bork); // expected-error {{use of undeclared identifier 'bork'}}
};

struct no_found_size_not_in_substruct {
  unsigned long flags;
  unsigned char size; // expected-note {{'size' declared here}}
  struct A {
    int dummy;
    int * ptr __sized_by_or_null(size); // expected-error {{'sized_by_or_null' field 'size' isn't within the same struct as the annotated pointer}}
  } a;
};

struct not_found_size_not_in_unnamed_substruct {
  unsigned char size; // expected-note {{'size' declared here}}
  struct {
    int dummy;
    int * ptr __sized_by_or_null(size); // expected-error {{'sized_by_or_null' field 'size' isn't within the same struct as the annotated pointer}}
  } a;
};

struct not_found_size_not_in_unnamed_substruct_2 {
  struct {
    unsigned char size; // expected-note {{'size' declared here}}
  };
  struct {
    int dummy;
    int * ptr __sized_by_or_null(size); // expected-error {{'sized_by_or_null' field 'size' isn't within the same struct as the annotated pointer}}
  } a;
};

struct not_found_size_in_other_unnamed_substruct {
  struct {
    unsigned char size;
  } a1;

  struct {
    int dummy;
    int * ptr __sized_by_or_null(size); // expected-error {{use of undeclared identifier 'size'}}
  };
};

struct not_found_size_in_other_substruct {
  struct _a1 {
    unsigned char size;
  } a1;

  struct {
    int dummy;
    int * ptr __sized_by_or_null(size); // expected-error {{use of undeclared identifier 'size'}}
  };
};

struct not_found_size_in_other_substruct_2 {
  struct _a2 {
    unsigned char size;
  } a2;

  int * ptr __sized_by_or_null(size); // expected-error {{use of undeclared identifier 'size'}}
};

struct not_found_suggest {
  int bork;
  struct bar **ptr __sized_by_or_null(blork); // expected-error {{use of undeclared identifier 'blork'}}
};

int global; // expected-note {{'global' declared here}}

struct found_outside_of_struct {
  int bork;
  struct bar ** ptr __sized_by_or_null(global); // expected-error {{field 'global' in 'sized_by_or_null' not inside structure}}
};

struct self_referrential {
  int bork;
  // immediate-error@+2{{use of undeclared identifier 'self'}}
  // late-error@+1{{'sized_by_or_null' only applies to pointers; did you mean to use 'counted_by'?}}
  struct bar *self[] __sized_by_or_null(self);
};

struct non_int_size {
  double dbl_size;
  struct bar ** ptr __sized_by_or_null(dbl_size); // expected-error {{'sized_by_or_null' requires a non-boolean integer type argument}}
};

struct array_of_ints_size {
  int integers[2];
  struct bar ** ptr __sized_by_or_null(integers); // expected-error {{'sized_by_or_null' requires a non-boolean integer type argument}}
};

struct not_a_c99_fam {
  int size;
  struct bar *non_c99_fam[0] __sized_by_or_null(size); // expected-error {{'sized_by_or_null' only applies to pointers; did you mean to use 'counted_by'?}}
};

struct annotated_with_anon_struct {
  unsigned long flags;
  struct {
    unsigned char size;
    int * ptr __sized_by_or_null(crount); // expected-error {{use of undeclared identifier 'crount'}}
  };
};

//==============================================================================
// __sized_by_or_null on a struct ptr with element type that has unknown size
//==============================================================================

struct size_unknown;
struct on_member_ptr_incomplete_ty_ty_pos {
  int size;
  struct size_unknown * ptr __sized_by_or_null(size);
};

struct on_member_ptr_incomplete_const_ty_ty_pos {
  int size;
  const struct size_unknown * ptr __sized_by_or_null(size);
};

struct on_member_ptr_void_ty_ty_pos {
  int size;
  void * ptr __sized_by_or_null(size);
};

typedef void(fn_ty)(int);

struct on_member_ptr_fn_ptr_ty {
  int size;
  fn_ty* * ptr __sized_by_or_null(size);
};

struct on_member_ptr_fn_ty {
  int size;
  // expected-error@+1{{'sized_by_or_null' cannot be applied to a pointer with pointee of unknown size because 'fn_ty' (aka 'void (int)') is a function type}}
  fn_ty * ptr __sized_by_or_null(size);
};
