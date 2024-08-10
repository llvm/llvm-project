// RUN: %clang_cc1 %s -ast-dump | FileCheck %s
// RUN: %clang_cc1 -fexperimental-late-parse-attributes %s -ast-dump | FileCheck %s

#define __sized_by_or_null(f)  __attribute__((sized_by_or_null(f)))

struct size_unknown;
struct size_known {
  int field;
};

//==============================================================================
// __sized_by_or_null on struct member pointer in decl attribute position
//==============================================================================

// CHECK-LABEL: RecordDecl {{.+}} struct on_member_pointer_complete_ty definition
// CHECK-NEXT: |-FieldDecl {{.+}} referenced count 'int'
// CHECK-NEXT: `-FieldDecl {{.+}} buf 'struct size_known * __sized_by_or_null(count)':'struct size_known *'
struct on_member_pointer_complete_ty {
  int count;
  struct size_known * buf __sized_by_or_null(count);
};

// CHECK-LABEL: RecordDecl {{.+}} struct on_pointer_anon_buf definition
// CHECK-NEXT:  |-FieldDecl {{.+}} referenced count 'int'
// CHECK-NEXT:  |-RecordDecl {{.+}} struct definition
// CHECK-NEXT:  | `-FieldDecl {{.+}} buf 'struct size_known * __sized_by_or_null(count)':'struct size_known *'
// CHECK-NEXT:  |-FieldDecl {{.+}} implicit 'struct on_pointer_anon_buf::(anonymous at [[ANON_STRUCT_PATH:.+]])'
// CHECK-NEXT:  `-IndirectFieldDecl {{.+}} implicit buf 'struct size_known * __sized_by_or_null(count)':'struct size_known *'
// CHECK-NEXT:    |-Field {{.+}} '' 'struct on_pointer_anon_buf::(anonymous at [[ANON_STRUCT_PATH]])'
// CHECK-NEXT:    `-Field {{.+}} 'buf' 'struct size_known * __sized_by_or_null(count)':'struct size_known *'
struct on_pointer_anon_buf {
  int count;
  struct {
    struct size_known *buf __sized_by_or_null(count);
  };
};

struct on_pointer_anon_count {
  struct {
    int count;
  };
  struct size_known *buf __sized_by_or_null(count);
};

//==============================================================================
// __sized_by_or_null on struct member pointer in type attribute position
//==============================================================================
// TODO: Correctly parse sized_by_or_null as a type attribute. Currently it is parsed
// as a declaration attribute

// CHECK-LABEL: RecordDecl {{.+}} struct on_member_pointer_complete_ty_ty_pos definition
// CHECK-NEXT:  |-FieldDecl {{.+}} referenced count 'int'
// CHECK-NEXT:  `-FieldDecl {{.+}} buf 'struct size_known * __sized_by_or_null(count)':'struct size_known *'
struct on_member_pointer_complete_ty_ty_pos {
  int count;
  struct size_known *__sized_by_or_null(count) buf;
};

// TODO: This should be forbidden but isn't due to sized_by_or_null being treated as a
// declaration attribute. The attribute ends up on the outer most pointer
// (allowed by sema) even though syntactically its supposed to be on the inner
// pointer (would not allowed by sema due to pointee being a function type).
// CHECK-LABEL: RecordDecl {{.+}} struct on_member_pointer_fn_ptr_ty_ty_pos_inner definition
// CHECK-NEXT:  |-FieldDecl {{.+}} referenced count 'int'
// CHECK-NEXT:  `-FieldDecl {{.+}} fn_ptr 'void (** __sized_by_or_null(count))(void)':'void (**)(void)'
struct on_member_pointer_fn_ptr_ty_ty_pos_inner {
  int count;
  void (* __sized_by_or_null(count) * fn_ptr)(void);
};

// FIXME: The generated AST here is wrong. The attribute should be on the inner
// pointer.
// CHECK-LABEL: RecordDecl {{.+}} struct on_nested_pointer_inner definition
// CHECK-NEXT:  |-FieldDecl {{.+}} referenced count 'int'
// CHECK-NEXT:  `-FieldDecl {{.+}} buf 'struct size_known ** __sized_by_or_null(count)':'struct size_known **'
struct on_nested_pointer_inner {
  int count;
  // TODO: This should be disallowed because in the `-fbounds-safety` model
  // `__sized_by_or_null` can only be nested when used in function parameters.
  struct size_known *__sized_by_or_null(count) *buf;
};

// CHECK-LABEL: RecordDecl {{.+}} struct on_nested_pointer_outer definition
// CHECK-NEXT:  |-FieldDecl {{.+}} referenced count 'int'
// CHECK-NEXT:  `-FieldDecl {{.+}} buf 'struct size_known ** __sized_by_or_null(count)':'struct size_known **'
struct on_nested_pointer_outer {
  int count;
  struct size_known **__sized_by_or_null(count) buf;
};

// CHECK-LABEL: RecordDecl {{.+}} struct on_pointer_anon_buf_ty_pos definition
// CHECK-NEXT:  |-FieldDecl {{.+}} referenced count 'int'
// CHECK-NEXT:  |-RecordDecl {{.+}} struct definition
// CHECK-NEXT:  | `-FieldDecl {{.+}} buf 'struct size_known * __sized_by_or_null(count)':'struct size_known *'
// CHECK-NEXT:  |-FieldDecl {{.+}} implicit 'struct on_pointer_anon_buf_ty_pos::(anonymous at [[ANON_STRUCT_PATH2:.+]])'
// CHECK-NEXT:  `-IndirectFieldDecl {{.+}} implicit buf 'struct size_known * __sized_by_or_null(count)':'struct size_known *'
// CHECK-NEXT:    |-Field {{.+}} '' 'struct on_pointer_anon_buf_ty_pos::(anonymous at [[ANON_STRUCT_PATH2]])'
// CHECK-NEXT:    `-Field {{.+}} 'buf' 'struct size_known * __sized_by_or_null(count)':'struct size_known *'
struct on_pointer_anon_buf_ty_pos {
  int count;
  struct {
    struct size_known * __sized_by_or_null(count) buf;
  };
};

// CHECK-LABEL: RecordDecl {{.+}} struct on_pointer_anon_count_ty_pos definition
// CHECK-NEXT:  |-RecordDecl {{.+}} struct definition
// CHECK-NEXT:  | `-FieldDecl {{.+}} count 'int'
// CHECK-NEXT:  |-FieldDecl {{.+}} implicit 'struct on_pointer_anon_count_ty_pos::(anonymous at [[ANON_STRUCT_PATH3:.+]])'
// CHECK-NEXT:  |-IndirectFieldDecl {{.+}} implicit referenced count 'int'
// CHECK-NEXT:  | |-Field {{.+}} '' 'struct on_pointer_anon_count_ty_pos::(anonymous at [[ANON_STRUCT_PATH3]])'
// CHECK-NEXT:  | `-Field {{.+}} 'count' 'int'
struct on_pointer_anon_count_ty_pos {
  struct {
    int count;
  };
  struct size_known *__sized_by_or_null(count) buf;
};
