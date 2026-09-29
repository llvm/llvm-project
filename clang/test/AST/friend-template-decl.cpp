// RUN: %clang_cc1 -std=c++20 -ast-print %s | FileCheck %s --check-prefix=PRINT
// RUN: %clang_cc1 -std=c++20 -ast-dump -ast-dump-filter C %s | FileCheck %s --check-prefix=DUMP
// RUN: %clang_cc1 -std=c++20 -ast-dump=json -ast-dump-filter C %s | FileCheck %s --check-prefix=JSON
// RUN: %clang_cc1 -std=c++20 -x c++-header -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++20 -x c++ -include-pch %t -ast-dump-all -ast-dump-filter C /dev/null | FileCheck %s --check-prefix=DUMP

namespace N {
template <class T> struct A {
  template <class U> struct B;
};

template <class V> struct C {
  template <class T> template <class U> friend struct A<T>::B;
};

template struct C<int>;
}

// PRINT:      template<> struct C<int> {
// PRINT-NEXT:     template <class T> template <class U> friend struct A<T>::B;

// DUMP:      ClassTemplateDecl {{.*}} C
// DUMP:      FriendTemplateDecl {{.*}} 'struct A<T>::B'
// DUMP-NEXT:   |-TemplateTypeParmDecl {{.*}} T
// DUMP-NEXT:   `-TemplateTypeParmDecl {{.*}} U
// DUMP:      ClassTemplateSpecializationDecl {{.*}} struct C definition
// DUMP-NOT:  <<<NULL>>>
// DUMP:      FriendTemplateDecl {{.*}} qualified
// DUMP-NEXT:   |-NestedNameSpecifier TypeSpec 'A<T>'
// DUMP-NEXT:   |-ClassTemplateDecl
// DUMP-NEXT:   |-TemplateTypeParmDecl {{.*}} T
// DUMP-NEXT:   `-TemplateTypeParmDecl {{.*}} U

// JSON:      "kind": "ClassTemplateSpecializationDecl",
// JSON:      "kind": "FriendTemplateDecl",
// JSON-NEXT: "loc": {
// JSON:      "templateName": "A<T>::B"
