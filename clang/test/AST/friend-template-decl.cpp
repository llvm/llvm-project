// RUN: %clang_cc1 -std=c++20 -ast-print %s | FileCheck %s --check-prefix=PRINT
// RUN: %clang_cc1 -std=c++20 -ast-dump -ast-dump-filter S %s | FileCheck %s --check-prefix=DUMP
// RUN: %clang_cc1 -std=c++20 -ast-dump=json -ast-dump-filter S %s | FileCheck %s --check-prefix=JSON

namespace N {
template <class T> struct A {
  template <class U> struct B;
};

template <class V> struct S {
  template <class T> template <class U> friend struct A<T>::B;
};

template struct S<int>;
}

// PRINT:      template<> struct S<int> {
// PRINT-NEXT:     template <class T> template <class U> friend struct A<T>::B;

// DUMP:      ClassTemplateDecl {{.*}} S
// DUMP:      FriendTemplateDecl {{.*}} 'struct A<T>::B'
// DUMP:      ClassTemplateSpecializationDecl {{.*}} struct S definition
// DUMP-NOT:  <<<NULL>>>
// DUMP:      FriendTemplateDecl {{.*}} qualified
// DUMP-NEXT:   |-NestedNameSpecifier TypeSpec 'A<T>'
// DUMP-NEXT:   `-ClassTemplateDecl

// JSON:      "kind": "ClassTemplateSpecializationDecl",
// JSON:      "kind": "FriendTemplateDecl",
// JSON-NEXT: "loc": {
// JSON:      "templateName": "A<T>::B"
