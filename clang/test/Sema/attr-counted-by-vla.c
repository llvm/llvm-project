// RUN: %clang_cc1 -fsyntax-only -verify=expected,immediate %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-late-parse-attributes %s -verify=expected,late

#define __counted_by(f)  __attribute__((counted_by(f)))

struct bar;

struct not_found {
  int count;
  struct bar *fam[] __counted_by(bork); // expected-error {{use of undeclared identifier 'bork'}}
};

struct no_found_count_not_in_substruct {
  unsigned long flags;
  unsigned char count; // expected-note {{'count' declared here}}
  struct A {
    int dummy;
    int array[] __counted_by(count); // expected-error {{'counted_by' field 'count' isn't within the same struct as the annotated flexible array}}
  } a;
};

struct not_found_count_not_in_unnamed_substruct {
  unsigned char count; // expected-note {{'count' declared here}}
  struct {
    int dummy;
    int array[] __counted_by(count); // expected-error {{'counted_by' field 'count' isn't within the same struct as the annotated flexible array}}
  } a;
};

struct not_found_count_not_in_unnamed_substruct_2 {
  struct {
    unsigned char count; // expected-note {{'count' declared here}}
  };
  struct {
    int dummy;
    int array[] __counted_by(count); // expected-error {{'counted_by' field 'count' isn't within the same struct as the annotated flexible array}}
  } a;
};

struct not_found_count_in_other_unnamed_substruct {
  struct {
    unsigned char count;
  } a1;

  struct {
    int dummy;
    int array[] __counted_by(count); // expected-error {{use of undeclared identifier 'count'}}
  };
};

struct not_found_count_in_other_substruct {
  struct _a1 {
    unsigned char count;
  } a1;

  struct {
    int dummy;
    int array[] __counted_by(count); // expected-error {{use of undeclared identifier 'count'}}
  };
};

struct not_found_count_in_other_substruct_2 {
  struct _a2 {
    unsigned char count;
  } a2;

  int array[] __counted_by(count); // expected-error {{use of undeclared identifier 'count'}}
};

struct not_found_suggest {
  int bork;
  struct bar *fam[] __counted_by(blork); // expected-error {{use of undeclared identifier 'blork'}}
};

int global; // expected-note {{'global' declared here}}

struct found_outside_of_struct {
  int bork;
  struct bar *fam[] __counted_by(global); // expected-error {{field 'global' in 'counted_by' not inside structure}}
};

struct self_referrential {
  int bork;
  // immediate-error@+2{{use of undeclared identifier 'self'}}
  // late-error@+1{{'counted_by' requires a non-boolean integer type argument}}
  struct bar *self[] __counted_by(self);
};

struct non_int_count {
  double dbl_count;
  struct bar *fam[] __counted_by(dbl_count); // expected-error {{'counted_by' requires a non-boolean integer type argument}}
};

struct array_of_ints_count {
  int integers[2];
  struct bar *fam[] __counted_by(integers); // expected-error {{'counted_by' requires a non-boolean integer type argument}}
};

struct not_a_fam {
  int count;
  struct bar *non_fam __counted_by(count); // ok
};

struct not_a_c99_fam {
  int count;
  struct bar *non_c99_fam[0] __counted_by(count); // expected-error {{'counted_by' on arrays only applies to C99 flexible array members}}
};

struct annotated_with_anon_struct {
  unsigned long flags;
  struct {
    unsigned char count;
    int array[] __counted_by(crount); // expected-error {{use of undeclared identifier 'crount'}}
  };
};

//==============================================================================
// __counted_by on a struct VLA with element type that has unknown size
//==============================================================================

struct size_unknown; // expected-note 2{{forward declaration of 'struct size_unknown'}}
struct on_member_arr_incomplete_ty_ty_pos {
  int count;
  // expected-error@+2{{'counted_by' only applies to pointers or C99 flexible array members}}
  // expected-error@+1{{array has incomplete element type 'struct size_unknown'}}
  struct size_unknown buf[] __counted_by(count);
};

struct on_member_arr_incomplete_const_ty_ty_pos {
  int count;
  // expected-error@+2{{'counted_by' only applies to pointers or C99 flexible array members}}
  // expected-error@+1{{array has incomplete element type 'const struct size_unknown'}}
  const struct size_unknown buf[] __counted_by(count);
};

struct on_member_arr_void_ty_ty_pos {
  int count;
  // expected-error@+2{{'counted_by' only applies to pointers or C99 flexible array members}}
  // expected-error@+1{{array has incomplete element type 'void'}}
  void buf[] __counted_by(count);
};

typedef void(fn_ty)(int);

struct on_member_arr_fn_ptr_ty {
  int count;
  // An Array of function pointers is allowed
  fn_ty* buf[] __counted_by(count);
};

struct on_member_arr_fn_ty {
  int count;
  // An array of functions is not allowed.
  // expected-error@+2{{'counted_by' only applies to pointers or C99 flexible array members}}
  // expected-error@+1{{'buf' declared as array of functions of type 'fn_ty' (aka 'void (int)')}}
  fn_ty buf[] __counted_by(count);
};


// `buffer_of_structs_with_unnannotated_vla`,
// `buffer_of_structs_with_annotated_vla`, and
// `buffer_of_const_structs_with_annotated_vla` are currently prevented because
// computing the size of `Arr` at runtime would require an O(N) walk of `Arr`
// elements to take into account the length of the VLA in each struct instance.

struct has_unannotated_VLA {
  int count;
  char buffer[];
};

struct has_annotated_VLA {
  int count;
  char buffer[] __counted_by(count);
};

struct buffer_of_structs_with_unnannotated_vla {
  int count;
  // Treating this as a warning is a temporary fix for existing attribute adopters. It **SHOULD BE AN ERROR**.
  // expected-warning@+1{{'counted_by' should not be applied to an array with element of unknown size because 'struct has_unannotated_VLA' is a struct type with a flexible array member. This will be an error in a future compiler version}}
  struct has_unannotated_VLA Arr[] __counted_by(count);
};


struct buffer_of_structs_with_annotated_vla {
  int count;
  // Treating this as a warning is a temporary fix for existing attribute adopters. It **SHOULD BE AN ERROR**.
  // expected-warning@+1{{'counted_by' should not be applied to an array with element of unknown size because 'struct has_annotated_VLA' is a struct type with a flexible array member. This will be an error in a future compiler version}}
  struct has_annotated_VLA Arr[] __counted_by(count);
};

struct buffer_of_const_structs_with_annotated_vla {
  int count;
  // Treating this as a warning is a temporary fix for existing attribute adopters. It **SHOULD BE AN ERROR**.
  // Make sure the `const` qualifier is printed when printing the element type.
  // expected-warning@+1{{'counted_by' should not be applied to an array with element of unknown size because 'const struct has_annotated_VLA' is a struct type with a flexible array member. This will be an error in a future compiler version}}
  const struct has_annotated_VLA Arr[] __counted_by(count);
};

//==============================================================================
// __counted_by on flexible array members in unions
//==============================================================================

// FAM in anonymous union with count in parent struct - OK
struct fam_in_anon_union_count_in_parent {
  int count;
  union {
    char a;
    char fam[] __counted_by(count);
  };
};

// FAM in named union - ERROR
union fam_in_named_union {
  int count;
  char fam[] __counted_by(count); // expected-error {{'counted_by' cannot be applied to a union member}}
};

// Both FAM and count in same anonymous union - ERROR (they share storage)
struct fam_and_count_in_same_anon_union {
  union {
    int count;
    char fam[] __counted_by(count); // expected-error {{'counted_by' cannot be applied to a union member}}
  };
};

// Count in anonymous union, FAM in parent struct - ERROR (count in union)
struct count_in_anon_union_fam_in_parent {
  union {
    int count;
    int x;
  };
  char fam[] __counted_by(count); // expected-error {{'counted_by' argument cannot refer to a union member}}
};

// FAM in nested anonymous struct inside anonymous union - OK
struct fam_in_nested_anon_struct_in_anon_union {
  int count;
  union {
    int a;
    struct {
      char fam[] __counted_by(count);
    };
  };
};

// Count in anonymous union, but hidden by struct - ERROR (count in union)
struct count_in_deep_anon_union {
  union {
    struct {
      int count;
    };
  };
  char fam[] __counted_by(count); // expected-error {{'counted_by' argument cannot refer to a union member}}
};
