// RUN: %clang_cc1 -triple x86_64-unknown-windows-msvc -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s
// REQUIRES: asserts

struct S {
  void *p;
};

struct S s[] = {
  { .p = (void *)0, },
};

extern struct S t __attribute__((__alias__("s")));

// CHECK: !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "t", scope: {{.*}}, entity: {{.*}}, file: {{.*}}, line: 12)
