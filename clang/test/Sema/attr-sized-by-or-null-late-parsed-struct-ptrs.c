// RUN: %clang_cc1 -fexperimental-late-parse-attributes -fsyntax-only -verify %s

#define __sized_by_or_null(f)  __attribute__((__sized_by_or_null__(f)))

struct size_unknown;
struct size_known {
  int field;
};

typedef void(*fn_ptr_ty)(void);

//==============================================================================
// __sized_by_or_null on struct member pointer in decl attribute position
//==============================================================================

struct on_member_pointer_complete_ty {
  struct size_known * buf __sized_by_or_null(size);
  int size;
};

struct on_member_pointer_incomplete_ty {
  struct size_unknown * buf __sized_by_or_null(size);
  int size;
};

struct on_member_pointer_const_incomplete_ty {
  const struct size_unknown * buf __sized_by_or_null(size);
  int size;
};

struct on_member_pointer_void_ty {
  void* buf __sized_by_or_null(size);
  int size;
};

struct on_member_pointer_fn_ptr_ty {
  // buffer of `size` function pointers is allowed
  void (**fn_ptr)(void) __sized_by_or_null(size);
  int size;
};


struct on_member_pointer_fn_ptr_ty_ptr_ty {
  // buffer of `size` function pointers is allowed
  fn_ptr_ty* fn_ptr __sized_by_or_null(size);
  int size;
};

struct on_member_pointer_fn_ty {
  // buffer of function(s) with size `size` is allowed
  // expected-error@+1{{'sized_by_or_null' cannot be applied to a pointer with pointee of unknown size because 'void (void)' is a function type}}
  void (*fn_ptr)(void) __sized_by_or_null(size);
  int size;
};

struct on_member_pointer_fn_ptr_ty_ty {
  // buffer of function(s) with size `size` is allowed
  // expected-error@+1{{'sized_by_or_null' cannot be applied to a pointer with pointee of unknown size because 'void (void)' is a function type}}
  fn_ptr_ty fn_ptr __sized_by_or_null(size);
  int size;
};

struct has_unannotated_vla {
  int size;
  int buffer[];
};

struct on_member_pointer_struct_with_vla {
  // expected-error@+1{{'sized_by_or_null' cannot be applied to a pointer with pointee of unknown size because 'struct has_unannotated_vla' is a struct type with a flexible array member}}
  struct has_unannotated_vla* objects __sized_by_or_null(size);
  int size;
};

struct has_annotated_vla {
  int size;
  // expected-error@+1{{'sized_by_or_null' only applies to pointers; did you mean to use 'counted_by'?}}
  int buffer[] __sized_by_or_null(size);
};

struct on_member_pointer_struct_with_annotated_vla {
  // expected-error@+1{{'sized_by_or_null' cannot be applied to a pointer with pointee of unknown size because 'struct has_annotated_vla' is a struct type with a flexible array member}}
  struct has_annotated_vla* objects __sized_by_or_null(size);
  int size;
};

struct on_pointer_anon_buf {
  // TODO: Support referring to parent scope
  struct {
    // expected-error@+1{{use of undeclared identifier 'size'}}
    struct size_known *buf __sized_by_or_null(size);
  };
  int size;
};

struct on_pointer_anon_count {
  struct size_known *buf __sized_by_or_null(size);
  struct {
    int size;
  };
};

//==============================================================================
// __sized_by_or_null on struct member pointer in type attribute position
//==============================================================================
// TODO: Correctly parse sized_by_or_null as a type attribute. Currently it is parsed
// as a declaration attribute and is **not** late parsed resulting in the `size`
// field being unavailable.

struct on_member_pointer_complete_ty_ty_pos {
  // TODO: Allow this
  // expected-error@+1{{use of undeclared identifier 'size'}}
  struct size_known *__sized_by_or_null(size) buf;
  int size;
};

struct on_member_pointer_incomplete_ty_ty_pos {
  // TODO: Allow this
  // expected-error@+1{{use of undeclared identifier 'size'}}
  struct size_unknown * __sized_by_or_null(size) buf;
  int size;
};

struct on_member_pointer_const_incomplete_ty_ty_pos {
  // TODO: Allow this
  // expected-error@+1{{use of undeclared identifier 'size'}}
  const struct size_unknown * __sized_by_or_null(size) buf;
  int size;
};

struct on_member_pointer_void_ty_ty_pos {
  // TODO: This should fail because the attribute is
  // on a pointer with the pointee being an incomplete type.
  // expected-error@+1{{use of undeclared identifier 'size'}}
  void *__sized_by_or_null(size) buf;
  int size;
};

// -

struct on_member_pointer_fn_ptr_ty_pos {
  // TODO: buffer of `size` function pointers should be allowed
  // but fails because this isn't late parsed.
  // expected-error@+1{{use of undeclared identifier 'size'}}
  void (** __sized_by_or_null(size) fn_ptr)(void);
  int size;
};

struct on_member_pointer_fn_ptr_ty_ptr_ty_pos {
  // TODO: buffer of `size` function pointers should be allowed
  // but fails because this isn't late parsed.
  // expected-error@+1{{use of undeclared identifier 'size'}}
  fn_ptr_ty* __sized_by_or_null(size) fn_ptr;
  int size;
};

struct on_member_pointer_fn_ty_ty_pos {
  // TODO: This should fail because the attribute is
  // on a pointer with the pointee being a function type.
  // expected-error@+1{{use of undeclared identifier 'size'}}
  void (* __sized_by_or_null(size) fn_ptr)(void);
  int size;
};

struct on_member_pointer_fn_ptr_ty_ty_pos {
  // TODO: buffer of `size` function pointers should be allowed
  // expected-error@+1{{use of undeclared identifier 'size'}}
  void (** __sized_by_or_null(size) fn_ptr)(void);
  int size;
};

struct on_member_pointer_fn_ptr_ty_typedef_ty_pos {
  // TODO: This should be allowed with sized_by_or_null.
  // expected-error@+1{{use of undeclared identifier 'size'}}
  fn_ptr_ty __sized_by_or_null(size) fn_ptr;
  int size;
};

struct on_member_pointer_fn_ptr_ty_ty_pos_inner {
  // TODO: This should be allowed with sized_by_or_null.
  // expected-error@+1{{use of undeclared identifier 'size'}}
  void (* __sized_by_or_null(size) * fn_ptr)(void);
  int size;
};

struct on_member_pointer_struct_with_vla_ty_pos {
  // TODO: This should be allowed with sized_by_or_null.
  // expected-error@+1{{use of undeclared identifier 'size'}}
  struct has_unannotated_vla *__sized_by_or_null(size) objects;
  int size;
};

struct on_member_pointer_struct_with_annotated_vla_ty_pos {
  // TODO: This should be allowed with sized_by_or_null.
  // expected-error@+1{{use of undeclared identifier 'size'}}
  struct has_annotated_vla* __sized_by_or_null(size) objects;
  int size;
};

struct on_nested_pointer_inner {
  // TODO: This should be disallowed because in the `-fbounds-safety` model
  // `__sized_by_or_null` can only be nested when used in function parameters.
  // expected-error@+1{{use of undeclared identifier 'size'}}
  struct size_known *__sized_by_or_null(size) *buf;
  int size;
};

struct on_nested_pointer_outer {
  // TODO: Allow this
  // expected-error@+1{{use of undeclared identifier 'size'}}
  struct size_known **__sized_by_or_null(size) buf;
  int size;
};

struct on_pointer_anon_buf_ty_pos {
  struct {
    // TODO: Support referring to parent scope
    // expected-error@+1{{use of undeclared identifier 'size'}}
    struct size_known * __sized_by_or_null(size) buf;
  };
  int size;
};

struct on_pointer_anon_count_ty_pos {
  // TODO: Allow this
  // expected-error@+1{{use of undeclared identifier 'size'}}
  struct size_known *__sized_by_or_null(size) buf;
  struct {
    int size;
  };
};

//==============================================================================
// __sized_by_or_null on struct non-pointer members
//==============================================================================

struct on_pod_ty {
  // expected-error-re@+1{{'sized_by_or_null' only applies to pointers{{$}}}}
  int wrong_ty __sized_by_or_null(size);
  int size;
};

struct on_void_ty {
  // expected-error-re@+2{{'sized_by_or_null' only applies to pointers{{$}}}}
  // expected-error@+1{{field has incomplete type 'void'}}
  void wrong_ty __sized_by_or_null(size);
  int size;
};
