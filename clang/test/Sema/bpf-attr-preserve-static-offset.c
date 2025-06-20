// RUN: %clang_cc1 -ast-dump -triple bpf-pc-linux-gnu %s | FileCheck %s

// The 'preserve_static_offset' attribute should be propagated to
// inline declarations (foo's 'b', 'bb', 'c' but not 'd').
//
// CHECK:      RecordDecl {{.*}} struct foo definition
// CHECK-NEXT:   BPFPreserveStaticOffsetAttr
// CHECK-NEXT:   FieldDecl {{.*}} a
// CHECK-NEXT:   RecordDecl {{.*}} struct definition
// CHECK-NEXT:     FieldDecl {{.*}} aa
// CHECK-NEXT:   FieldDecl {{.*}} b
// CHECK-NEXT: RecordDecl {{.*}} union bar definition
// CHECK-NEXT:   BPFPreserveStaticOffsetAttr
// CHECK-NEXT:   FieldDecl {{.*}} a
// CHECK-NEXT:   FieldDecl {{.*}} b

struct foo {
  int a;
  struct {
    int aa;
  } b;
} __attribute__((preserve_static_offset));

union bar {
  int a;
  long b;
} __attribute__((preserve_static_offset));
