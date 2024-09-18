// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -verify %s
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -ast-dump-all %s | FileCheck %s
// expected-no-diagnostics

#pragma clang module build bounds_safety
module bounds_safety {}
#pragma clang module contents
#pragma clang module begin bounds_safety
struct Test {
  int count;
  int fam[] __attribute__((counted_by(count)));
};
#pragma clang module end
#pragma clang module endbuild

#pragma clang module import bounds_safety

struct Test *p;

// CHECK: |-RecordDecl {{.*}}bounds_safety.map:4:1, line:7:1> line:4:8 imported in bounds_safety <undeserialized declarations> struct Test definition
// CHECK: | |-FieldDecl {{.*}} imported in bounds_safety referenced count 'int'
// CHECK: | `-FieldDecl {{.*}} imported in bounds_safety fam 'int[] __counted_by(count)':'int[]'

// CHECK: |-ImportDecl {{.*}}bounds-safety-attributed-type.c:17:22> col:22 implicit bounds_safety
// CHECK: |-RecordDecl {{.*}} struct Test
// CHECK: `-VarDecl {{.*}} p 'struct Test *'
