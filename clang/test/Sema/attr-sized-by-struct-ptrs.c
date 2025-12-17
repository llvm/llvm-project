// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fexperimental-late-parse-attributes -fsyntax-only -verify %s

#define __sized_by(f)  __attribute__((sized_by(f)))
#define __counted_by(f)  __attribute__((counted_by(f)))

struct size_unknown;
struct size_known {
  int field;
};

typedef void(*fn_ptr_ty)(void);

//==============================================================================
// __sized_by on struct member pointer in decl attribute position
//==============================================================================

struct on_member_pointer_complete_ty {
  int size;
  struct size_known * buf __sized_by(size);
};

struct on_member_pointer_incomplete_ty {
  int size;
  struct size_unknown * buf __sized_by(size);
};

struct on_member_pointer_const_incomplete_ty {
  int size;
  const struct size_unknown * buf __sized_by(size);
};

struct on_member_pointer_void_ty {
  int size;
  void* buf __sized_by(size);
};

struct on_member_pointer_fn_ptr_ty {
  int size;
  // buffer of function pointers with size `size` is allowed
  void (**fn_ptr)(void) __sized_by(size);
};

struct on_member_pointer_fn_ptr_ty_ptr_ty {
  int size;
  // buffer of function pointers with size `size` is allowed
  fn_ptr_ty* fn_ptr __sized_by(size);
};

struct on_member_pointer_fn_ty {
  int size;
  // buffer of functions with size `size` is allowed
  // expected-error@+1{{'sized_by' cannot be applied to a pointer with pointee of unknown size because 'void (void)' is a function type}}
  void (*fn_ptr)(void) __sized_by(size);
};

struct on_member_pointer_fn_ptr_ty_ty {
  int size;
  // buffer of functions with size `size` is allowed
  // expected-error@+1{{'sized_by' cannot be applied to a pointer with pointee of unknown size because 'void (void)' is a function type}}
  fn_ptr_ty fn_ptr __sized_by(size);
};

struct has_unannotated_vla {
  int count;
  int buffer[];
};

struct on_member_pointer_struct_with_vla {
  int size;
  // we know the size so this is fine for tracking size, however indexing would be an issue
  // expected-error@+1{{'sized_by' cannot be applied to a pointer with pointee of unknown size because 'struct has_unannotated_vla' is a struct type with a flexible array member}}
  struct has_unannotated_vla* objects __sized_by(size);
};

struct has_annotated_vla {
  int count;
  int buffer[] __counted_by(count);
};

struct on_member_pointer_struct_with_annotated_vla {
  int size;
  // we know the size so this is fine for tracking size, however indexing would be an issue
  // expected-error@+1{{'sized_by' cannot be applied to a pointer with pointee of unknown size because 'struct has_annotated_vla' is a struct type with a flexible array member}}
  struct has_annotated_vla* objects __sized_by(size);
};

struct on_pointer_anon_buf {
  int size;
  struct {
    struct size_known *buf __sized_by(size);
  };
};

struct on_pointer_anon_size {
  struct {
    int size;
  };
  struct size_known *buf __sized_by(size);
};

//==============================================================================
// __sized_by on struct member pointer in type attribute position
//==============================================================================
// TODO: Correctly parse sized_by as a type attribute. Currently it is parsed
// as a declaration attribute

struct on_member_pointer_complete_ty_ty_pos {
  int size;
  struct size_known *__sized_by(size) buf;
};

struct on_member_pointer_incomplete_ty_ty_pos {
  int size;
  struct size_unknown * __sized_by(size) buf;
};

struct on_member_pointer_const_incomplete_ty_ty_pos {
  int size;
  const struct size_unknown * __sized_by(size) buf;
};

struct on_member_pointer_void_ty_ty_pos {
  int size;
  void *__sized_by(size) buf;
};

// -

struct on_member_pointer_fn_ptr_ty_pos {
  int size;
  // buffer of `size` function pointers is allowed
  void (** __sized_by(size) fn_ptr)(void);
};

struct on_member_pointer_fn_ptr_ty_ptr_ty_pos {
  int size;
  // buffer of `size` function pointers is allowed
  fn_ptr_ty* __sized_by(size) fn_ptr;
};

struct on_member_pointer_fn_ty_ty_pos {
  int size;
  // expected-error@+1{{'sized_by' cannot be applied to a pointer with pointee of unknown size because 'void (void)' is a function type}}
  void (* __sized_by(size) fn_ptr)(void);
};

struct on_member_pointer_fn_ptr_ty_ty_pos {
  int size;
  // expected-error@+1{{'sized_by' cannot be applied to a pointer with pointee of unknown size because 'void (void)' is a function type}}
  fn_ptr_ty __sized_by(size) fn_ptr;
};

// TODO: This should be forbidden but isn't due to sized_by being treated
// as a declaration attribute.
struct on_member_pointer_fn_ptr_ty_ty_pos_inner {
  int size;
  void (* __sized_by(size) * fn_ptr)(void);
};

struct on_member_pointer_struct_with_vla_ty_pos {
  int size;
  // expected-error@+1{{'sized_by' cannot be applied to a pointer with pointee of unknown size because 'struct has_unannotated_vla' is a struct type with a flexible array member}}
  struct has_unannotated_vla *__sized_by(size) objects;
};

struct on_member_pointer_struct_with_annotated_vla_ty_pos {
  int size;
  // expected-error@+1{{'sized_by' cannot be applied to a pointer with pointee of unknown size because 'struct has_annotated_vla' is a struct type with a flexible array member}}
  struct has_annotated_vla* __sized_by(size) objects;
};

struct on_nested_pointer_inner {
  // TODO: This should be disallowed because in the `-fbounds-safety` model
  // `__sized_by` can only be nested when used in function parameters.
  int size;
  struct size_known *__sized_by(size) *buf;
};

struct on_nested_pointer_outer {
  int size;
  struct size_known **__sized_by(size) buf;
};

struct on_pointer_anon_buf_ty_pos {
  int size;
  struct {
    struct size_known * __sized_by(size) buf;
  };
};

struct on_pointer_anon_size_ty_pos {
  struct {
    int size;
  };
  struct size_known *__sized_by(size) buf;
};

//==============================================================================
// __sized_by on struct non-pointer members
//==============================================================================

struct on_pod_ty {
  int size;
  // expected-error-re@+1{{'sized_by' only applies to pointers{{$}}}}
  int wrong_ty __sized_by(size);
};

struct on_void_ty {
  int size;
  // expected-error-re@+2{{'sized_by' only applies to pointers{{$}}}}
  // expected-error@+1{{field has incomplete type 'void'}}
  void wrong_ty __sized_by(size);
};

struct on_member_array_complete_ty {
  int size;
  // expected-error@+1{{'sized_by' only applies to pointers; did you mean to use 'counted_by'?}}
  struct size_known array[] __sized_by(size);
};
