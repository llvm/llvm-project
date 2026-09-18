// RUN: not %clang_cc1 %s -ast-dump | FileCheck %s
// RUN: not %clang_cc1 -fexperimental-late-parse-attributes %s -ast-dump | FileCheck %s

// A rejected counted_by-family attribute leaves the field's type unchanged,
// qualifiers included.

#define __counted_by(f)  __attribute__((counted_by(f)))

struct size_known {
  int field;
};

// CHECK-LABEL: RecordDecl {{.+}} union on_pointer_in_union definition
// CHECK-NEXT: |-FieldDecl {{.+}} count 'int'
// CHECK-NEXT: `-FieldDecl {{.+}} buf 'struct size_known *volatile'
union on_pointer_in_union {
  int count;
  struct size_known *__counted_by(count) volatile buf;
};

typedef struct size_known *SKP;

// CHECK-LABEL: RecordDecl {{.+}} struct on_pointer_non_int_count definition
// CHECK-NEXT: |-FieldDecl {{.+}} count 'double'
// CHECK-NEXT: `-FieldDecl {{.+}} buf 'const SKP':'struct size_known *const'
struct on_pointer_non_int_count {
  double count;
  const SKP __counted_by(count) buf;
};
