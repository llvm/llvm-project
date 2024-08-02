// RUN: %clang_cc1 -fexperimental-late-parse-attributes -fsyntax-only -verify %s

#define __counted_by_or_null(f)  __attribute__((counted_by_or_null(f)))
#define __counted_by(f)  __attribute__((counted_by(f)))

struct size_unknown;
struct size_known {
  int field;
};

typedef void(*fn_ptr_ty)(void);

//==============================================================================
// __counted_by_or_null on struct member pointer in decl attribute position
//==============================================================================

struct on_member_pointer_complete_ty {
  struct size_known * buf __counted_by_or_null(count);
  int count;
};

struct on_member_pointer_incomplete_ty {
  struct size_unknown * buf __counted_by_or_null(count); // expected-error{{'counted_by_or_null' cannot be applied to a pointer with pointee of unknown size because 'struct size_unknown' is an incomplete type}}
  int count;
};

struct on_member_pointer_const_incomplete_ty {
  // expected-error@+1{{'counted_by_or_null' cannot be applied to a pointer with pointee of unknown size because 'const struct size_unknown' is an incomplete type}}
  const struct size_unknown * buf __counted_by_or_null(count);
  int count;
};

struct on_member_pointer_void_ty {
  void* buf __counted_by_or_null(count); // expected-error{{'counted_by_or_null' cannot be applied to a pointer with pointee of unknown size because 'void' is an incomplete type}}
  int count;
};

struct on_member_pointer_fn_ptr_ty {
  // buffer of `count` function pointers is allowed
  void (**fn_ptr)(void) __counted_by_or_null(count);
  int count;
};


struct on_member_pointer_fn_ptr_ty_ptr_ty {
  // buffer of `count` function pointers is allowed
  fn_ptr_ty* fn_ptr __counted_by_or_null(count);
  int count;
};

struct on_member_pointer_fn_ty {
  // buffer of `count` functions is not allowed
  // expected-error@+1{{'counted_by_or_null' cannot be applied to a pointer with pointee of unknown size because 'void (void)' is a function type}}
  void (*fn_ptr)(void) __counted_by_or_null(count);
  int count;
};

struct on_member_pointer_fn_ptr_ty_ty {
  // buffer of `count` functions is not allowed
  // expected-error@+1{{'counted_by_or_null' cannot be applied to a pointer with pointee of unknown size because 'void (void)' is a function type}}
  fn_ptr_ty fn_ptr __counted_by_or_null(count);
  int count;
};

struct has_unannotated_vla {
  int count;
  int buffer[];
};

struct on_member_pointer_struct_with_vla {
  // expected-error@+1{{'counted_by_or_null' cannot be applied to a pointer with pointee of unknown size because 'struct has_unannotated_vla' is a struct type with a flexible array member}}
  struct has_unannotated_vla* objects __counted_by_or_null(count);
  int count;
};

struct has_annotated_vla {
  int count;
  int buffer[] __counted_by(count);
};

// Currently prevented because computing the size of `objects` at runtime would
// require an O(N) walk of `objects` to take into account the length of the VLA
// in each struct instance.
struct on_member_pointer_struct_with_annotated_vla {
  // expected-error@+1{{'counted_by_or_null' cannot be applied to a pointer with pointee of unknown size because 'struct has_annotated_vla' is a struct type with a flexible array member}}
  struct has_annotated_vla* objects __counted_by_or_null(count);
  int count;
};

struct on_pointer_anon_buf {
  // TODO: Support referring to parent scope
  struct {
    // expected-error@+1{{use of undeclared identifier 'count'}}
    struct size_known *buf __counted_by_or_null(count);
  };
  int count;
};

struct on_pointer_anon_count {
  struct size_known *buf __counted_by_or_null(count);
  struct {
    int count;
  };
};

//==============================================================================
// __counted_by_or_null on struct member pointer in type attribute position
//==============================================================================
// TODO: Correctly parse counted_by_or_null as a type attribute. Currently it is parsed
// as a declaration attribute and is **not** late parsed resulting in the `count`
// field being unavailable.

struct on_member_pointer_complete_ty_ty_pos {
  // TODO: Allow this
  // expected-error@+1{{use of undeclared identifier 'count'}}
  struct size_known *__counted_by_or_null(count) buf;
  int count;
};

struct on_member_pointer_incomplete_ty_ty_pos {
  // TODO: Allow this
  // expected-error@+1{{use of undeclared identifier 'count'}}
  struct size_unknown * __counted_by_or_null(count) buf;
  int count;
};

struct on_member_pointer_const_incomplete_ty_ty_pos {
  // TODO: Allow this
  // expected-error@+1{{use of undeclared identifier 'count'}}
  const struct size_unknown * __counted_by_or_null(count) buf;
  int count;
};

struct on_member_pointer_void_ty_ty_pos {
  // TODO: This should fail because the attribute is
  // on a pointer with the pointee being an incomplete type.
  // expected-error@+1{{use of undeclared identifier 'count'}}
  void *__counted_by_or_null(count) buf;
  int count;
};

// -

struct on_member_pointer_fn_ptr_ty_pos {
  // TODO: buffer of `count` function pointers should be allowed
  // but fails because this isn't late parsed.
  // expected-error@+1{{use of undeclared identifier 'count'}}
  void (** __counted_by_or_null(count) fn_ptr)(void);
  int count;
};

struct on_member_pointer_fn_ptr_ty_ptr_ty_pos {
  // TODO: buffer of `count` function pointers should be allowed
  // but fails because this isn't late parsed.
  // expected-error@+1{{use of undeclared identifier 'count'}}
  fn_ptr_ty* __counted_by_or_null(count) fn_ptr;
  int count;
};

struct on_member_pointer_fn_ty_ty_pos {
  // TODO: This should fail because the attribute is
  // on a pointer with the pointee being a function type.
  // expected-error@+1{{use of undeclared identifier 'count'}}
  void (* __counted_by_or_null(count) fn_ptr)(void);
  int count;
};

struct on_member_pointer_fn_ptr_ty_ty_pos {
  // TODO: buffer of `count` function pointers should be allowed
  // expected-error@+1{{use of undeclared identifier 'count'}}
  void (** __counted_by_or_null(count) fn_ptr)(void);
  int count;
};

struct on_member_pointer_fn_ptr_ty_typedef_ty_pos {
  // TODO: This should fail because the attribute is
  // on a pointer with the pointee being a function type.
  // expected-error@+1{{use of undeclared identifier 'count'}}
  fn_ptr_ty __counted_by_or_null(count) fn_ptr;
  int count;
};

struct on_member_pointer_fn_ptr_ty_ty_pos_inner {
  // TODO: This should fail because the attribute is
  // on a pointer with the pointee being a function type.
  // expected-error@+1{{use of undeclared identifier 'count'}}
  void (* __counted_by_or_null(count) * fn_ptr)(void);
  int count;
};

struct on_member_pointer_struct_with_vla_ty_pos {
  // TODO: This should fail because the attribute is
  // on a pointer with the pointee being a struct type with a VLA.
  // expected-error@+1{{use of undeclared identifier 'count'}}
  struct has_unannotated_vla *__counted_by_or_null(count) objects;
  int count;
};

struct on_member_pointer_struct_with_annotated_vla_ty_pos {
  // TODO: This should fail because the attribute is
  // on a pointer with the pointee being a struct type with a VLA.
  // expected-error@+1{{use of undeclared identifier 'count'}}
  struct has_annotated_vla* __counted_by_or_null(count) objects;
  int count;
};

struct on_nested_pointer_inner {
  // TODO: This should be disallowed because in the `-fbounds-safety` model
  // `__counted_by_or_null` can only be nested when used in function parameters.
  // expected-error@+1{{use of undeclared identifier 'count'}}
  struct size_known *__counted_by_or_null(count) *buf;
  int count;
};

struct on_nested_pointer_outer {
  // TODO: Allow this
  // expected-error@+1{{use of undeclared identifier 'count'}}
  struct size_known **__counted_by_or_null(count) buf;
  int count;
};

struct on_pointer_anon_buf_ty_pos {
  struct {
    // TODO: Support referring to parent scope
    // expected-error@+1{{use of undeclared identifier 'count'}}
    struct size_known * __counted_by_or_null(count) buf;
  };
  int count;
};

struct on_pointer_anon_count_ty_pos {
  // TODO: Allow this
  // expected-error@+1{{use of undeclared identifier 'count'}}
  struct size_known *__counted_by_or_null(count) buf;
  struct {
    int count;
  };
};

//==============================================================================
// __counted_by_or_null on struct non-pointer members
//==============================================================================

struct on_pod_ty {
  // expected-error-re@+1{{'counted_by_or_null' only applies to pointers{{$}}}}
  int wrong_ty __counted_by_or_null(count);
  int count;
};

struct on_void_ty {
  // expected-error-re@+2{{'counted_by_or_null' only applies to pointers{{$}}}}
  // expected-error@+1{{field has incomplete type 'void'}}
  void wrong_ty __counted_by_or_null(count);
  int count;
};
