// RUN: %clang_cc1 -fsyntax-only -verify %s
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
  int count;
  struct size_known * buf __counted_by_or_null(count);
};

struct on_member_pointer_incomplete_ty {
  int count;
  // expected-error@+1{{'counted_by_or_null' cannot be applied to a pointer with pointee of unknown size because 'struct size_unknown' is an incomplete type}}
  struct size_unknown * buf __counted_by_or_null(count);
};

struct on_member_pointer_const_incomplete_ty {
  int count;
  // expected-error@+1{{'counted_by_or_null' cannot be applied to a pointer with pointee of unknown size because 'const struct size_unknown' is an incomplete type}}
  const struct size_unknown * buf __counted_by_or_null(count);
};

struct on_member_pointer_void_ty {
  int count;
  // expected-error@+1{{'counted_by_or_null' cannot be applied to a pointer with pointee of unknown size because 'void' is an incomplete type}}
  void* buf __counted_by_or_null(count);
};

struct on_member_pointer_fn_ptr_ty {
  int count;
  // buffer of `count` function pointers is allowed
  void (**fn_ptr)(void) __counted_by_or_null(count);
};

struct on_member_pointer_fn_ptr_ty_ptr_ty {
  int count;
  // buffer of `count` function pointers is allowed
  fn_ptr_ty* fn_ptr __counted_by_or_null(count);
};

struct on_member_pointer_fn_ty {
  int count;
  // buffer of `count` functions is not allowed
  // expected-error@+1{{'counted_by_or_null' cannot be applied to a pointer with pointee of unknown size because 'void (void)' is a function type}}
  void (*fn_ptr)(void) __counted_by_or_null(count);
};

struct on_member_pointer_fn_ptr_ty_ty {
  int count;
  // buffer of `count` functions is not allowed
  // expected-error@+1{{'counted_by_or_null' cannot be applied to a pointer with pointee of unknown size because 'void (void)' is a function type}}
  fn_ptr_ty fn_ptr __counted_by_or_null(count);
};

struct has_unannotated_vla {
  int count;
  int buffer[];
};

struct on_member_pointer_struct_with_vla {
  int count;
  // expected-error@+1{{'counted_by_or_null' cannot be applied to a pointer with pointee of unknown size because 'struct has_unannotated_vla' is a struct type with a flexible array member}}
  struct has_unannotated_vla* objects __counted_by_or_null(count);
};

struct has_annotated_vla {
  int count;
  int buffer[] __counted_by(count);
};

// Currently prevented because computing the size of `objects` at runtime would
// require an O(N) walk of `objects` to take into account the length of the VLA
// in each struct instance.
struct on_member_pointer_struct_with_annotated_vla {
  int count;
  // expected-error@+1{{'counted_by_or_null' cannot be applied to a pointer with pointee of unknown size because 'struct has_annotated_vla' is a struct type with a flexible array member}}
  struct has_annotated_vla* objects __counted_by_or_null(count);
};

struct on_pointer_anon_buf {
  int count;
  struct {
    struct size_known *buf __counted_by_or_null(count);
  };
};

struct on_pointer_anon_count {
  struct {
    int count;
  };
  struct size_known *buf __counted_by_or_null(count);
};

//==============================================================================
// __counted_by_or_null on struct member pointer in type attribute position
//==============================================================================
// TODO: Correctly parse counted_by_or_null as a type attribute. Currently it is parsed
// as a declaration attribute

struct on_member_pointer_complete_ty_ty_pos {
  int count;
  struct size_known *__counted_by_or_null(count) buf;
};

struct on_member_pointer_incomplete_ty_ty_pos {
  int count;
  // expected-error@+1{{'counted_by_or_null' cannot be applied to a pointer with pointee of unknown size because 'struct size_unknown' is an incomplete type}}
  struct size_unknown * __counted_by_or_null(count) buf;
};

struct on_member_pointer_const_incomplete_ty_ty_pos {
  int count;
  // expected-error@+1{{'counted_by_or_null' cannot be applied to a pointer with pointee of unknown size because 'const struct size_unknown' is an incomplete type}}
  const struct size_unknown * __counted_by_or_null(count) buf;
};

struct on_member_pointer_void_ty_ty_pos {
  int count;
  // expected-error@+1{{'counted_by_or_null' cannot be applied to a pointer with pointee of unknown size because 'void' is an incomplete type}}
  void *__counted_by_or_null(count) buf;
};

// -

struct on_member_pointer_fn_ptr_ty_pos {
  int count;
  // buffer of `count` function pointers is allowed
  void (** __counted_by_or_null(count) fn_ptr)(void);
};

struct on_member_pointer_fn_ptr_ty_ptr_ty_pos {
  int count;
  // buffer of `count` function pointers is allowed
  fn_ptr_ty* __counted_by_or_null(count) fn_ptr;
};

struct on_member_pointer_fn_ty_ty_pos {
  int count;
  // buffer of `count` functions is not allowed
  // expected-error@+1{{'counted_by_or_null' cannot be applied to a pointer with pointee of unknown size because 'void (void)' is a function type}}
  void (* __counted_by_or_null(count) fn_ptr)(void);
};

struct on_member_pointer_fn_ptr_ty_ty_pos {
  int count;
  // buffer of `count` functions is not allowed
  // expected-error@+1{{'counted_by_or_null' cannot be applied to a pointer with pointee of unknown size because 'void (void)' is a function type}}
  fn_ptr_ty __counted_by_or_null(count) fn_ptr;
};

// TODO: This should be forbidden but isn't due to counted_by_or_null being treated
// as a declaration attribute.
struct on_member_pointer_fn_ptr_ty_ty_pos_inner {
  int count;
  void (* __counted_by_or_null(count) * fn_ptr)(void);
};

struct on_member_pointer_struct_with_vla_ty_pos {
  int count;
  // expected-error@+1{{'counted_by_or_null' cannot be applied to a pointer with pointee of unknown size because 'struct has_unannotated_vla' is a struct type with a flexible array member}}
  struct has_unannotated_vla *__counted_by_or_null(count) objects;
};

// Currently prevented because computing the size of `objects` at runtime would
// require an O(N) walk of `objects` to take into account the length of the VLA
// in each struct instance.
struct on_member_pointer_struct_with_annotated_vla_ty_pos {
  int count;
  // expected-error@+1{{counted_by_or_null' cannot be applied to a pointer with pointee of unknown size because 'struct has_annotated_vla' is a struct type with a flexible array member}}
  struct has_annotated_vla* __counted_by_or_null(count) objects;
};

struct on_nested_pointer_inner {
  // TODO: This should be disallowed because in the `-fbounds-safety` model
  // `__counted_by_or_null` can only be nested when used in function parameters.
  int count;
  struct size_known *__counted_by_or_null(count) *buf;
};

struct on_nested_pointer_outer {
  int count;
  struct size_known **__counted_by_or_null(count) buf;
};

struct on_pointer_anon_buf_ty_pos {
  int count;
  struct {
    struct size_known * __counted_by_or_null(count) buf;
  };
};

struct on_pointer_anon_count_ty_pos {
  struct {
    int count;
  };
  struct size_known *__counted_by_or_null(count) buf;
};

//==============================================================================
// __counted_by_or_null on struct non-pointer members
//==============================================================================

struct on_pod_ty {
  int count;
  // expected-error-re@+1{{'counted_by_or_null' only applies to pointers{{$}}}}
  int wrong_ty __counted_by_or_null(count);
};

struct on_void_ty {
  int count;
  // expected-error-re@+2{{'counted_by_or_null' only applies to pointers{{$}}}}
  // expected-error@+1{{field has incomplete type 'void'}}
  void wrong_ty __counted_by_or_null(count);
};
