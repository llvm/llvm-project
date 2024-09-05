// Without serialization:
// RUN: %clang_cc1 -ast-dump %s | FileCheck %s
//
// With serialization:
// RUN: %clang_cc1 -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -include-pch %t -ast-dump-all  /dev/null | FileCheck %s

struct S {
  // CHECK:      CXXMethodDecl {{.*}} a 'void ()' delete
  // CHECK-NEXT:   delete message: StringLiteral {{.*}} "foo"
  void a() = delete("foo");

  // CHECK:      FunctionTemplateDecl {{.*}} b
  // CHECK-NEXT:   TemplateTypeParmDecl
  // CHECK-NEXT:   CXXMethodDecl {{.*}} b 'void ()' delete
  // CHECK-NEXT:     delete message: StringLiteral {{.*}} "bar"
  template <typename>
  void b() = delete("bar");
};

// CHECK:      FunctionDecl {{.*}} c 'void ()' delete
// CHECK-NEXT:   delete message: StringLiteral {{.*}} "baz"
void c() = delete("baz");
