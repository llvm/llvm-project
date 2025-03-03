// RUN: %clang_cc1 -fexperimental-late-parse-attributes %s -ast-dump | FileCheck %s

#define __counted_by_or_null(f)  __attribute__((counted_by_or_null(f)))

struct size_known {
  int field;
};

//==============================================================================
// __counted_by_or_null on struct member pointer in decl attribute position
//==============================================================================

struct on_member_pointer_complete_ty {
  struct size_known *buf __counted_by_or_null(count);
  int count;
};
// CHECK-LABEL: struct on_member_pointer_complete_ty definition
// CHECK-NEXT: |-FieldDecl {{.*}} buf 'struct size_known * __counted_by_or_null(count)':'struct size_known *'
// CHECK-NEXT: `-FieldDecl {{.*}} referenced count 'int'

struct on_pointer_anon_count {
  struct size_known *buf __counted_by_or_null(count);
  struct {
    int count;
  };
};

// CHECK-LABEL: struct on_pointer_anon_count definition
// CHECK-NEXT:  |-FieldDecl {{.*}} buf 'struct size_known * __counted_by_or_null(count)':'struct size_known *'
// CHECK-NEXT:  |-RecordDecl {{.*}} struct definition
// CHECK-NEXT:  | `-FieldDecl {{.*}} count 'int'
// CHECK-NEXT:  |-FieldDecl {{.*}} implicit 'struct on_pointer_anon_count::(anonymous at {{.*}})'
// CHECK-NEXT:  `-IndirectFieldDecl {{.*}} implicit referenced count 'int'
// CHECK-NEXT:    |-Field {{.*}} field_index 1 'struct on_pointer_anon_count::(anonymous at {{.*}})'
// CHECK-NEXT:    `-Field {{.*}} 'count' 'int'

//==============================================================================
// __counted_by_or_null on struct member pointer in type attribute position
//==============================================================================
// TODO: Correctly parse counted_by_or_null as a type attribute. Currently it is parsed
// as a declaration attribute and is **not** late parsed resulting in the `count`
// field being unavailable.
//
// See `clang/test/Sema/attr-counted-by-late-parsed-struct-ptrs.c` for test
// cases.
