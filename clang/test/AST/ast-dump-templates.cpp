// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++20 -ast-dump=json %s | FileCheck --check-prefix=JSON %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++20 -ast-print %s > %t
// RUN: FileCheck < %t %s -check-prefix=CHECK1
// RUN: FileCheck < %t %s -check-prefix=CHECK2
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++20 -ast-dump %s | FileCheck --check-prefix=DUMP %s

// Test with serialization:
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++20 -emit-pch -o %t %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -x c++ -std=c++20 -include-pch %t \
// RUN: -ast-dump-all /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck --strict-whitespace --check-prefix=DUMP %s

template <int X, typename Y, int Z = 5>
struct foo {
  int constant;
  foo() {}
  Y getSum() { return Y(X + Z); }
};

template <int A, typename B>
B bar() {
  return B(A);
}

void baz() {
  int x = bar<5, int>();
  int y = foo<5, int>().getSum();
  double z = foo<2, double, 3>().getSum();
}

// Template definition - foo
// CHECK1: template <int X, typename Y, int Z = 5> struct foo {
// CHECK2: template <int X, typename Y, int Z = 5> struct foo {

// Template instantiation - foo
// Since the order of instantiation may vary during runs, run FileCheck twice
// to make sure each instantiation is in the correct spot.
// CHECK1: template<> struct foo<5, int, 5> {
// CHECK2: template<> struct foo<2, double, 3> {

// Template definition - bar
// CHECK1: template <int A, typename B> B bar()
// CHECK2: template <int A, typename B> B bar()

// Template instantiation - bar
// CHECK1: template<> int bar<5, int>()
// CHECK2: template<> int bar<5, int>()

// CHECK1-LABEL: template <typename ...T> struct A {
// CHECK1-NEXT:    template <T ...x[3]> struct B {
template <typename ...T> struct A {
  template <T ...x[3]> struct B {};
};

// CHECK1-LABEL: template <typename ...T> void f() {
// CHECK1-NEXT:    A<T[3]...> a;
template <typename ...T> void f() {
  A<T[3]...> a;
}

namespace test2 {
void func(int);
void func(float);
template<typename T>
void tmpl() {
  func(T());
}

// DUMP: UnresolvedLookupExpr {{.*}} <col:3> '<overloaded function type>' lvalue (ADL) = 'func'

// JSON-LABEL: "name": "test2",
// JSON:       "kind": "UnresolvedLookupExpr",
// JSON-NEXT:  "range": {
// JSON-NEXT:    "begin": {
// JSON-NEXT:      "offset": {{.*}},
// JSON-NEXT:      "col": 3,
// JSON-NEXT:      "tokLen": {{.*}}
// JSON-NEXT:    },
// JSON-NEXT:    "end": {
// JSON-NEXT:      "offset": {{.*}},
// JSON-NEXT:      "col": 3,
// JSON-NEXT:      "tokLen": {{.*}}
// JSON-NEXT:    }
// JSON-NEXT:  },
// JSON-NEXT:  "type": {
// JSON-NEXT:    "qualType": "<overloaded function type>"
// JSON-NEXT:  },
// JSON-NEXT:  "valueCategory": "lvalue",
// JSON-NEXT:  "usesADL": true,
// JSON-NEXT:  "name": "func",
}

namespace test3 {
  template<typename T> struct A {};
  template<typename T> A(T) -> A<int>;
  // CHECK1: template <typename T> A(T) -> A<int>;
}

namespace test4 {
template <unsigned X, auto A>
struct foo {
  static void fn();
};

// Prints using an "integral" template argument. Test that this correctly
// includes the type for the auto argument and omits it for the fixed
// type/unsigned argument (see
// TemplateParameterList::shouldIncludeTypeForArgument)
// CHECK1: {{^    }}template<> struct foo<0, 0L> {
// CHECK1: {{^    }}void test(){{ }}{
// CHECK1: {{^        }}foo<0, 0 + 0L>::fn();
void test() {
  foo<0, 0 + 0L>::fn();
}

// Prints using an "expression" template argument. This renders based on the way
// the user wrote the arguments (including that + expression) - so it's not
// powered by the shouldIncludeTypeForArgument functionality.
// Not sure if this it's intentional that these two specializations are rendered
// differently in this way.
// CHECK1: {{^    }}template<> struct foo<1, 0 + 0L> {
template struct foo<1, 0 + 0L>;
}

namespace test5 {
template<long> void f() {}
void (*p)() = f<0>;
template<unsigned = 0> void f() {}
void (*q)() = f<>;
// Not perfect - this code in the dump would be ambiguous, but it's the best we
// can do to differentiate these two implicit specializations.
// CHECK1: template<> void f<0L>()
// CHECK1: template<> void f<0U>()
}

namespace test6 {
template <class D>
constexpr bool C = true;

template <class Key>
void func() {
  C<Key>;
// DUMP:      UnresolvedLookupExpr {{.*}} '<dependent type>' lvalue (no ADL) = 'C'
// DUMP-NEXT: `-TemplateArgument type 'Key'
// DUMP-NEXT:   `-TemplateTypeParmType {{.*}} 'Key' dependent depth 0 index 0
// DUMP-NEXT:     `-TemplateTypeParm {{.*}} 'Key'

// JSON-LABEL: "name": "test6",
// JSON:       "kind": "UnresolvedLookupExpr",
// JSON:       "type": {
// JSON-NEXT:    "qualType": "<dependent type>"
// JSON-NEXT:  },
// JSON-NEXT:  "valueCategory": "lvalue",
// JSON-NEXT:  "usesADL": false,
// JSON-NEXT:  "name": "C",
// JSON-NEXT:  "lookups": [
// JSON-NEXT:    {
// JSON-NEXT:      "id": {{.*}},
// JSON-NEXT:      "kind": "VarTemplateDecl",
// JSON-NEXT:      "name": "C"
// JSON-NEXT:    }
// JSON-NEXT:  ],
// JSON-NEXT:  "inner": [
// JSON-NEXT:    {
// JSON-NEXT:      "kind": "TemplateArgument",
// JSON-NEXT:      "type": {
// JSON-NEXT:        "qualType": "Key"
// JSON-NEXT:      },
// JSON-NEXT:      "inner": [
// JSON-NEXT:        {
// JSON-NEXT:          "id": {{.*}},
// JSON-NEXT:          "kind": "TemplateTypeParmType",
// JSON-NEXT:          "type": {
// JSON-NEXT:            "qualType": "Key"
// JSON-NEXT:          },
// JSON-NEXT:          "isDependent": true,
// JSON-NEXT:          "isInstantiationDependent": true,
// JSON-NEXT:          "depth": 0,
// JSON-NEXT:          "index": 0,
// JSON-NEXT:          "decl": {
// JSON-NEXT:            "id": {{.*}},
// JSON-NEXT:            "kind": "TemplateTypeParmDecl",
// JSON-NEXT:            "name": "Key"
// JSON-NEXT:          }
}
}

namespace test7 {
  template <template<class> class TT> struct A {};
  template <class...> class B {};
  template struct A<B>;
// DUMP-LABEL: NamespaceDecl {{.*}} test7 external-linkage{{$}}
// DUMP:       ClassTemplateDecl 0x{{.+}} A external-linkage{{$}}
// DUMP-NEXT:  |-TemplateTemplateParmDecl
// DUMP-NEXT:  | `-TemplateTypeParmDecl
// DUMP-NEXT:  |-CXXRecordDecl 0x[[TEST7_PAT:[^ ]+]] {{.+}} struct A definition
// DUMP:       ClassTemplateSpecializationDecl {{.*}} struct A definition external-linkage instantiated_from 0x[[TEST7_PAT]] explicit_instantiation_definition strict-pack-match{{$}}

// JSON-LABEL: "name": "test7",
// JSON:         "kind": "ClassTemplateSpecializationDecl",
// JSON:         "name": "A",
// JSON-NEXT:    "tagUsed": "struct",
// JSON-NEXT:    "completeDefinition": true,
// JSON-NEXT:    "strict-pack-match": true,
} // namespce test7

namespace test8 {
template<_Complex int x>
struct pr126341;
template<>
struct pr126341<{1, 2}>;
// DUMP-LABEL: NamespaceDecl {{.*}} test8 external-linkage{{$}}
// DUMP-NEXT:  |-ClassTemplateDecl {{.*}} pr126341
// DUMP:       `-ClassTemplateSpecializationDecl {{.*}} pr126341
// DUMP:         `-TemplateArgument structural value '1+2i'

// JSON-LABEL: "name": "test8",
// JSON:         "kind": "ClassTemplateSpecializationDecl",
// JSON:         "name": "pr126341",
// JSON-NEXT:    "tagUsed": "struct",
// JSON-NEXT:    "inner": [
// JSON-NEXT:      {
// JSON-NEXT:        "kind": "TemplateArgument",
// JSON-NEXT:        "value": "1+2i"
} // namespace test8

namespace TestMemberPointerPartialSpec {
  template <class> struct A;
  template <class T1, class T2> struct A<T1 T2::*>;
// DUMP-LABEL: NamespaceDecl {{.+}} TestMemberPointerPartialSpec external-linkage{{$}}
// DUMP:       ClassTemplatePartialSpecializationDecl {{.*}} struct A
// DUMP-NEXT:  |-TemplateArgument type 'type-parameter-0-0 type-parameter-0-1::*'
// DUMP-NEXT:  | `-MemberPointerType {{.+}} 'type-parameter-0-0 type-parameter-0-1::*' dependent
// DUMP-NEXT:  |   |-TemplateTypeParmType {{.+}} 'type-parameter-0-1' dependent depth 0 index 1
// DUMP-NEXT:  |   `-TemplateTypeParmType {{.+}} 'type-parameter-0-0' dependent depth 0 index 0

// JSON-LABEL: "name": "TestMemberPointerPartialSpec",
// JSON:         "kind": "ClassTemplatePartialSpecializationDecl"
// JSON:         "name": "A",
// JSON-NEXT:    "tagUsed": "struct",
// JSON-NEXT:    "inner": [
// JSON-NEXT:      {
// JSON-NEXT:        "kind": "TemplateArgument",
// JSON-NEXT:        "type": {
// JSON-NEXT:          "qualType": "type-parameter-0-0 type-parameter-0-1::*"
// JSON-NEXT:        },
// JSON-NEXT:        "inner": [
// JSON-NEXT:          {
// JSON-NEXT:            "id": {{.*}},
// JSON-NEXT:            "kind": "MemberPointerType",
// JSON-NEXT:            "type": {
// JSON-NEXT:              "qualType": "type-parameter-0-0 type-parameter-0-1::*"
// JSON-NEXT:            },
// JSON-NEXT:            "isDependent": true,
// JSON-NEXT:            "isInstantiationDependent": true,
// JSON-NEXT:            "isData": true,
// JSON-NEXT:            "inner": [
// JSON-NEXT:              {
// JSON-NEXT:                "id": {{.*}},
// JSON-NEXT:                "kind": "TemplateTypeParmType",
// JSON-NEXT:                "type": {
// JSON-NEXT:                  "qualType": "type-parameter-0-1"
// JSON-NEXT:                },
// JSON-NEXT:                "isDependent": true,
// JSON-NEXT:                "isInstantiationDependent": true,
// JSON-NEXT:                "depth": 0,
// JSON-NEXT:                "index": 1,
// JSON-NEXT:                "decl": {
// JSON-NEXT:                  "id": "0x0"
// JSON-NEXT:                }
// JSON-NEXT:              },
// JSON-NEXT:              {
// JSON-NEXT:                "id": {{.*}},
// JSON-NEXT:                "kind": "TemplateTypeParmType",
// JSON-NEXT:                "type": {
// JSON-NEXT:                  "qualType": "type-parameter-0-0"
// JSON-NEXT:                },
// JSON-NEXT:                "isDependent": true,
// JSON-NEXT:                "isInstantiationDependent": true,
// JSON-NEXT:                "depth": 0,
// JSON-NEXT:                "index": 0,
// JSON-NEXT:                "decl": {
// JSON-NEXT:                  "id": "0x0"
// JSON-NEXT:                }
} // namespace TestMemberPointerPartialSpec

namespace TestDependentMemberPointer {
  template <class U> struct A {
    using X = int U::*;
    using Y = int U::test::*;
    using Z = int U::template V<int>::*;
  };
// DUMP-LABEL: NamespaceDecl {{.+}} TestDependentMemberPointer external-linkage{{$}}
// DUMP:       |-TypeAliasDecl {{.+}} X 'int U::*'{{$}}
// DUMP-NEXT:  | `-MemberPointerType {{.+}} 'int U::*' dependent
// DUMP-NEXT:  |   |-TemplateTypeParmType {{.+}} 'U' dependent depth 0 index 0
// DUMP-NEXT:  |   | `-TemplateTypeParm {{.+}} 'U'
// DUMP-NEXT:  |   `-BuiltinType {{.+}} 'int'
// DUMP-NEXT:  |-TypeAliasDecl {{.+}} Y 'int U::test::*'{{$}}
// DUMP-NEXT:  | `-MemberPointerType {{.+}} 'int U::test::*' dependent
// DUMP-NEXT:  |   |-DependentNameType {{.+}} 'U::test' dependent
// DUMP-NEXT:  |   `-BuiltinType {{.+}} 'int'
// DUMP-NEXT:  `-TypeAliasDecl {{.+}} Z 'int U::template V<int>::*'{{$}}
// DUMP-NEXT:    `-MemberPointerType {{.+}} 'int U::template V<int>::*' dependent
// DUMP-NEXT:      |-TemplateSpecializationType {{.+}} 'U::template V<int>' dependent
// DUMP-NEXT:      | |-name: 'U::template V':'type-parameter-0-0::template V' dependent
// DUMP-NEXT:      | | `-NestedNameSpecifier TypeSpec 'U'
// DUMP-NEXT:      | `-TemplateArgument type 'int'
// DUMP-NEXT:      `-BuiltinType {{.+}} 'int'

// JSON-LABEL: "name": "TestDependentMemberPointer"
// JSON:         "kind": "TypeAliasDecl",
// JSON:         "name": "X",
// JSON-NEXT:    "type": {
// JSON-NEXT:      "qualType": "int U::*"
// JSON-NEXT:    },
// JSON-NEXT:    "inner": [
// JSON-NEXT:      {
// JSON-NEXT:        "id": {{.+}},
// JSON-NEXT:        "kind": "MemberPointerType",
// JSON-NEXT:        "type": {
// JSON-NEXT:          "qualType": "int U::*"
// JSON-NEXT:        },
// JSON-NEXT:        "isDependent": true,
// JSON-NEXT:        "isInstantiationDependent": true,
// JSON-NEXT:        "isData": true,
// JSON-NEXT:        "inner": [
// JSON-NEXT:          {
// JSON-NEXT:            "id": {{.+}},
// JSON-NEXT:            "kind": "TemplateTypeParmType",
// JSON-NEXT:            "type": {
// JSON-NEXT:              "qualType": "U"
// JSON-NEXT:            },
// JSON-NEXT:            "isDependent": true,
// JSON-NEXT:            "isInstantiationDependent": true,
// JSON-NEXT:            "depth": 0,
// JSON-NEXT:            "index": 0,
// JSON-NEXT:            "decl": {
// JSON-NEXT:              "id": {{.+}},
// JSON-NEXT:              "kind": "TemplateTypeParmDecl",
// JSON-NEXT:              "name": "U"
// JSON-NEXT:            }
// JSON-NEXT:          },
// JSON-NEXT:          {
// JSON-NEXT:            "id": {{.+}},
// JSON-NEXT:            "kind": "BuiltinType",
// JSON-NEXT:            "type": {
// JSON-NEXT:              "qualType": "int"
// JSON-NEXT:            }
// JSON-NEXT:          }
// JSON:         "kind": "TypeAliasDecl",
// JSON:         "name": "Y",
// JSON-NEXT:    "type": {
// JSON-NEXT:      "qualType": "int U::test::*"
// JSON-NEXT:    },
// JSON-NEXT:    "inner": [
// JSON-NEXT:      {
// JSON-NEXT:        "id": {{.+}},
// JSON-NEXT:        "kind": "MemberPointerType",
// JSON-NEXT:        "type": {
// JSON-NEXT:          "qualType": "int U::test::*"
// JSON-NEXT:        },
// JSON-NEXT:        "isDependent": true,
// JSON-NEXT:        "isInstantiationDependent": true,
// JSON-NEXT:        "isData": true,
// JSON-NEXT:        "inner": [
// JSON-NEXT:          {
// JSON-NEXT:            "id": {{.+}},
// JSON-NEXT:            "kind": "DependentNameType",
// JSON-NEXT:            "type": {
// JSON-NEXT:              "qualType": "U::test"
// JSON-NEXT:            },
// JSON-NEXT:            "isDependent": true,
// JSON-NEXT:            "isInstantiationDependent": true
// JSON-NEXT:          },
// JSON-NEXT:          {
// JSON-NEXT:            "id": {{.+}},
// JSON-NEXT:            "kind": "BuiltinType",
// JSON-NEXT:            "type": {
// JSON-NEXT:              "qualType": "int"
// JSON-NEXT:            }
// JSON-NEXT:          }
// JSON:         "kind": "TypeAliasDecl",
// JSON:         "name": "Z",
// JSON-NEXT:    "type": {
// JSON-NEXT:      "qualType": "int U::template V<int>::*"
// JSON-NEXT:    },
// JSON-NEXT:    "inner": [
// JSON-NEXT:      {
// JSON-NEXT:        "id": {{.+}},
// JSON-NEXT:        "kind": "MemberPointerType",
// JSON-NEXT:        "type": {
// JSON-NEXT:          "qualType": "int U::template V<int>::*"
// JSON-NEXT:        },
// JSON-NEXT:        "isDependent": true,
// JSON-NEXT:        "isInstantiationDependent": true,
// JSON-NEXT:        "isData": true,
// JSON-NEXT:        "inner": [
// JSON-NEXT:          {
// JSON-NEXT:            "id": {{.+}},
// JSON-NEXT:            "kind": "TemplateSpecializationType",
// JSON-NEXT:            "type": {
// JSON-NEXT:              "qualType": "U::template V<int>"
// JSON-NEXT:            },
// JSON-NEXT:            "isDependent": true,
// JSON-NEXT:            "isInstantiationDependent": true,
// JSON-NEXT:            "templateName": "U::template V",
// JSON-NEXT:            "inner": [
// JSON-NEXT:              {
// JSON-NEXT:                "kind": "TemplateArgument",
// JSON-NEXT:                "type": {
// JSON-NEXT:                  "qualType": "int"
// JSON-NEXT:                },
// JSON-NEXT:                "inner": [
// JSON-NEXT:                  {
// JSON-NEXT:                    "id": {{.+}},
// JSON-NEXT:                    "kind": "BuiltinType",
// JSON-NEXT:                    "type": {
// JSON-NEXT:                      "qualType": "int"
// JSON-NEXT:                    }
// JSON-NEXT:                  }
// JSON-NEXT:                ]
// JSON-NEXT:              }
// JSON-NEXT:            ]
// JSON-NEXT:          },
// JSON-NEXT:          {
// JSON-NEXT:            "id": {{.+}},
// JSON-NEXT:            "kind": "BuiltinType",
// JSON-NEXT:            "type": {
// JSON-NEXT:              "qualType": "int"
// JSON-NEXT:            }
// JSON-NEXT:          }
} // namespace TestDependentMemberPointer

namespace TestPartialSpecNTTP {
// DUMP-LABEL: NamespaceDecl {{.+}} TestPartialSpecNTTP external-linkage{{$}}
// JSON-LABEL: "name": "TestPartialSpecNTTP"
  template <class TA1, bool TA2> struct Template1 {};
  template <class TB1, bool TB2> struct Template2 {};

  template <class U1, bool U2, bool U3>
  struct Template2<Template1<U1, U2>, U3> {};
// DUMP:      ClassTemplatePartialSpecializationDecl {{.+}} struct Template2
// DUMP:      |-TemplateArgument type 'TestPartialSpecNTTP::Template1<type-parameter-0-0, value-parameter-0-1>'
// DUMP-NEXT: | `-TemplateSpecializationType {{.+}} 'TestPartialSpecNTTP::Template1<type-parameter-0-0, value-parameter-0-1>' dependent
// DUMP-NEXT: |   |-name: 'TestPartialSpecNTTP::Template1'
// DUMP-NEXT: |   | `-ClassTemplateDecl {{.+}} Template1
// DUMP-NEXT: |   |-TemplateArgument type 'type-parameter-0-0'
// DUMP-NEXT: |   | `-TemplateTypeParmType {{.+}} 'type-parameter-0-0' dependent depth 0 index 0
// DUMP-NEXT: |   `-TemplateArgument expr canonical 'value-parameter-0-1'
// DUMP-NEXT: |     `-DeclRefExpr {{.+}} 'bool' NonTypeTemplateParm {{.+}} 'U2' 'bool'
// DUMP-NEXT: |-TemplateArgument expr canonical 'value-parameter-0-2'
// DUMP-NEXT: | `-DeclRefExpr {{.+}} 'bool' NonTypeTemplateParm {{.+}} 'U3' 'bool'
// DUMP-NEXT: |-TemplateTypeParmDecl {{.+}} referenced class depth 0 index 0 U1
// DUMP-NEXT: |-NonTypeTemplateParmDecl {{.+}} referenced 'bool' depth 0 index 1 U2
// DUMP-NEXT: |-NonTypeTemplateParmDecl {{.+}} referenced 'bool' depth 0 index 2 U3
// DUMP-NEXT: `-CXXRecordDecl {{.+}} implicit struct Template2

// JSON:      "kind": "ClassTemplatePartialSpecializationDecl",
// JSON:      "name": "Template2",
// JSON:      "tagUsed": "struct",
// JSON:      "inner": [
// JSON-NEXT:   {
// JSON-NEXT:     "kind": "TemplateArgument",
// JSON-NEXT:     "type": {
// JSON-NEXT:       "qualType": "TestPartialSpecNTTP::Template1<type-parameter-0-0, value-parameter-0-1>"
// JSON-NEXT:     },
// JSON-NEXT:     "inner": [
// JSON-NEXT:       {
// JSON-NEXT:         "id": {{.+}},
// JSON-NEXT:         "kind": "TemplateSpecializationType",
// JSON-NEXT:         "type": {
// JSON-NEXT:           "qualType": "TestPartialSpecNTTP::Template1<type-parameter-0-0, value-parameter-0-1>"
// JSON-NEXT:         },
// JSON-NEXT:         "isDependent": true,
// JSON-NEXT:         "isInstantiationDependent": true,
// JSON-NEXT:         "templateName": "TestPartialSpecNTTP::Template1",
// JSON-NEXT:         "inner": [
// JSON-NEXT:           {
// JSON-NEXT:             "kind": "TemplateArgument",
// JSON-NEXT:             "type": {
// JSON-NEXT:               "qualType": "type-parameter-0-0"
// JSON-NEXT:             },
// JSON-NEXT:             "inner": [
// JSON-NEXT:               {
// JSON-NEXT:                 "id": {{.+}},
// JSON-NEXT:                 "kind": "TemplateTypeParmType",
// JSON-NEXT:                 "type": {
// JSON-NEXT:                   "qualType": "type-parameter-0-0"
// JSON-NEXT:                 },
// JSON-NEXT:                 "isDependent": true,
// JSON-NEXT:                 "isInstantiationDependent": true,
// JSON-NEXT:                 "depth": 0,
// JSON-NEXT:                 "index": 0,
// JSON-NEXT:                 "decl": {
// JSON-NEXT:                   "id": "0x0"
// JSON-NEXT:                 }
// JSON-NEXT:               }
// JSON-NEXT:             ]
// JSON-NEXT:           },
// JSON-NEXT:           {
// JSON-NEXT:             "kind": "TemplateArgument",
// JSON-NEXT:             "isExpr": true,
// JSON-NEXT:             "isCanonical": true,
// JSON-NEXT:             "inner": [
// JSON-NEXT:               {
// JSON-NEXT:                 "id": {{.+}},
// JSON-NEXT:                 "kind": "DeclRefExpr",
// JSON:                      "type": {
// JSON-NEXT:                   "qualType": "bool"
// JSON-NEXT:                 },
// JSON-NEXT:                 "valueCategory": "prvalue",
// JSON-NEXT:                 "referencedDecl": {
// JSON-NEXT:                   "id": {{.+}},
// JSON-NEXT:                   "kind": "NonTypeTemplateParmDecl",
// JSON-NEXT:                   "name": "U2",
// JSON-NEXT:                   "type": {
// JSON-NEXT:                     "qualType": "bool"
// JSON:          "kind": "TemplateArgument",
// JSON-NEXT:     "isExpr": true,
// JSON-NEXT:     "isCanonical": true,
// JSON-NEXT:     "inner": [
// JSON-NEXT:       {
// JSON-NEXT:         "id": {{.+}},
// JSON-NEXT:         "kind": "DeclRefExpr",
// JSON:              "type": {
// JSON-NEXT:           "qualType": "bool"
// JSON-NEXT:         },
// JSON-NEXT:         "valueCategory": "prvalue",
// JSON-NEXT:         "referencedDecl": {
// JSON-NEXT:           "id": {{.+}},
// JSON-NEXT:           "kind": "NonTypeTemplateParmDecl",
// JSON-NEXT:           "name": "U3",
// JSON-NEXT:           "type": {
// JSON-NEXT:             "qualType": "bool"
// JSON-NEXT:           }
// JSON-NEXT:         }
// JSON-NEXT:       }
// JSON-NEXT:     ]
// JSON-NEXT:   },
// JSON-NEXT:   {
// JSON-NEXT:     "id": {{.+}},
// JSON-NEXT:     "kind": "TemplateTypeParmDecl",
// JSON:          "isReferenced": true,
// JSON-NEXT:     "name": "U1",
// JSON-NEXT:     "tagUsed": "class",
// JSON-NEXT:     "depth": 0,
// JSON-NEXT:     "index": 0
// JSON-NEXT:   },
// JSON-NEXT:   {
// JSON-NEXT:     "id": {{.+}},
// JSON-NEXT:     "kind": "NonTypeTemplateParmDecl",
// JSON:          "isReferenced": true,
// JSON-NEXT:     "name": "U2",
// JSON-NEXT:     "type": {
// JSON-NEXT:       "qualType": "bool"
// JSON-NEXT:     },
// JSON-NEXT:     "depth": 0,
// JSON-NEXT:     "index": 1
// JSON-NEXT:   },
// JSON-NEXT:   {
// JSON-NEXT:     "id": {{.+}},
// JSON-NEXT:     "kind": "NonTypeTemplateParmDecl",
// JSON:          "isReferenced": true,
// JSON-NEXT:     "name": "U3",
// JSON-NEXT:     "type": {
// JSON-NEXT:       "qualType": "bool"
// JSON-NEXT:     },
// JSON-NEXT:     "depth": 0,
// JSON-NEXT:     "index": 2
// JSON-NEXT:   },
// JSON-NEXT:   {
// JSON-NEXT:     "id": {{.+}},
// JSON-NEXT:     "kind": "CXXRecordDecl",
// JSON:          "isImplicit": true,
// JSON-NEXT:     "name": "Template2",
// JSON-NEXT:     "tagUsed": "struct"
// JSON-NEXT:   }
// JSON-NEXT: ]

  template <typename U1, bool U3, bool U2>
  struct Template2<Template1<U1, U2>, U3> {};
// DUMP:      ClassTemplatePartialSpecializationDecl {{.+}} struct Template2 definition external-linkage explicit_specialization
// DUMP:      |-TemplateArgument type 'TestPartialSpecNTTP::Template1<type-parameter-0-0, value-parameter-0-2>'
// DUMP-NEXT: | `-TemplateSpecializationType {{.+}} 'TestPartialSpecNTTP::Template1<type-parameter-0-0, value-parameter-0-2>' dependent
// DUMP-NEXT: |   |-name: 'TestPartialSpecNTTP::Template1'
// DUMP-NEXT: |   | `-ClassTemplateDecl {{.+}} Template1
// DUMP-NEXT: |   |-TemplateArgument type 'type-parameter-0-0'
// DUMP-NEXT: |   | `-TemplateTypeParmType {{.+}} 'type-parameter-0-0' dependent depth 0 index 0
// DUMP-NEXT: |   `-TemplateArgument expr canonical 'value-parameter-0-2'
// DUMP-NEXT: |     `-DeclRefExpr {{.+}} 'bool' NonTypeTemplateParm {{.+}} 'U2' 'bool'
// DUMP-NEXT: |-TemplateArgument expr canonical 'value-parameter-0-1'
// DUMP-NEXT: | `-DeclRefExpr {{.+}} 'bool' NonTypeTemplateParm {{.+}} 'U3' 'bool'
// DUMP-NEXT: |-TemplateTypeParmDecl {{.+}} referenced typename depth 0 index 0 U1
// DUMP-NEXT: |-NonTypeTemplateParmDecl {{.+}} referenced 'bool' depth 0 index 1 U3
// DUMP-NEXT: |-NonTypeTemplateParmDecl {{.+}} referenced 'bool' depth 0 index 2 U2
// DUMP-NEXT: `-CXXRecordDecl {{.+}} implicit struct Template2

// JSON:      "kind": "ClassTemplatePartialSpecializationDecl",
// JSON:      "name": "Template2",
// JSON:      "tagUsed": "struct",
// JSON:      "inner": [
// JSON-NEXT:   {
// JSON-NEXT:     "kind": "TemplateArgument",
// JSON-NEXT:     "type": {
// JSON-NEXT:       "qualType": "TestPartialSpecNTTP::Template1<type-parameter-0-0, value-parameter-0-2>"
// JSON-NEXT:     },
// JSON-NEXT:     "inner": [
// JSON-NEXT:       {
// JSON-NEXT:         "id": {{.+}},
// JSON-NEXT:         "kind": "TemplateSpecializationType",
// JSON-NEXT:         "type": {
// JSON-NEXT:           "qualType": "TestPartialSpecNTTP::Template1<type-parameter-0-0, value-parameter-0-2>"
// JSON-NEXT:         },
// JSON-NEXT:         "isDependent": true,
// JSON-NEXT:         "isInstantiationDependent": true,
// JSON-NEXT:         "templateName": "TestPartialSpecNTTP::Template1",
// JSON-NEXT:         "inner": [
// JSON-NEXT:           {
// JSON-NEXT:             "kind": "TemplateArgument",
// JSON-NEXT:             "type": {
// JSON-NEXT:               "qualType": "type-parameter-0-0"
// JSON-NEXT:             },
// JSON-NEXT:             "inner": [
// JSON-NEXT:               {
// JSON-NEXT:                 "id": {{.+}},
// JSON-NEXT:                 "kind": "TemplateTypeParmType",
// JSON-NEXT:                 "type": {
// JSON-NEXT:                   "qualType": "type-parameter-0-0"
// JSON-NEXT:                 },
// JSON-NEXT:                 "isDependent": true,
// JSON-NEXT:                 "isInstantiationDependent": true,
// JSON-NEXT:                 "depth": 0,
// JSON-NEXT:                 "index": 0,
// JSON-NEXT:                 "decl": {
// JSON-NEXT:                   "id": "0x0"
// JSON-NEXT:                 }
// JSON-NEXT:               }
// JSON-NEXT:             ]
// JSON-NEXT:           },
// JSON-NEXT:           {
// JSON-NEXT:             "kind": "TemplateArgument",
// JSON-NEXT:             "isExpr": true,
// JSON-NEXT:             "isCanonical": true,
// JSON-NEXT:             "inner": [
// JSON-NEXT:               {
// JSON-NEXT:                 "id": {{.+}},
// JSON-NEXT:                 "kind": "DeclRefExpr",
// JSON:                      "type": {
// JSON-NEXT:                   "qualType": "bool"
// JSON-NEXT:                 },
// JSON-NEXT:                 "valueCategory": "prvalue",
// JSON-NEXT:                 "referencedDecl": {
// JSON-NEXT:                   "id": {{.+}},
// JSON-NEXT:                   "kind": "NonTypeTemplateParmDecl",
// JSON-NEXT:                   "name": "U2",
// JSON-NEXT:                   "type": {
// JSON-NEXT:                     "qualType": "bool"
// JSON:          "kind": "TemplateArgument",
// JSON-NEXT:     "isExpr": true,
// JSON-NEXT:     "isCanonical": true,
// JSON-NEXT:     "inner": [
// JSON-NEXT:       {
// JSON-NEXT:         "id": {{.+}},
// JSON-NEXT:         "kind": "DeclRefExpr",
// JSON:              "type": {
// JSON-NEXT:           "qualType": "bool"
// JSON-NEXT:         },
// JSON-NEXT:         "valueCategory": "prvalue",
// JSON-NEXT:         "referencedDecl": {
// JSON-NEXT:           "id": {{.+}},
// JSON-NEXT:           "kind": "NonTypeTemplateParmDecl",
// JSON-NEXT:           "name": "U3",
// JSON-NEXT:           "type": {
// JSON-NEXT:             "qualType": "bool"
// JSON-NEXT:           }
// JSON-NEXT:         }
// JSON-NEXT:       }
// JSON-NEXT:     ]
// JSON-NEXT:   },
// JSON-NEXT:   {
// JSON-NEXT:     "id": {{.+}},
// JSON-NEXT:     "kind": "TemplateTypeParmDecl",
// JSON:          "isReferenced": true,
// JSON-NEXT:     "name": "U1",
// JSON-NEXT:     "tagUsed": "typename",
// JSON-NEXT:     "depth": 0,
// JSON-NEXT:     "index": 0
// JSON-NEXT:   },
// JSON-NEXT:   {
// JSON-NEXT:     "id": {{.+}},
// JSON-NEXT:     "kind": "NonTypeTemplateParmDecl",
// JSON:          "isReferenced": true,
// JSON-NEXT:     "name": "U3",
// JSON-NEXT:     "type": {
// JSON-NEXT:       "qualType": "bool"
// JSON-NEXT:     },
// JSON-NEXT:     "depth": 0,
// JSON-NEXT:     "index": 1
// JSON-NEXT:   },
// JSON-NEXT:   {
// JSON-NEXT:     "id": {{.+}},
// JSON-NEXT:     "kind": "NonTypeTemplateParmDecl",
// JSON:          "isReferenced": true,
// JSON-NEXT:     "name": "U2",
// JSON-NEXT:     "type": {
// JSON-NEXT:       "qualType": "bool"
// JSON-NEXT:     },
// JSON-NEXT:     "depth": 0,
// JSON-NEXT:     "index": 2
// JSON-NEXT:   },
// JSON-NEXT:   {
// JSON-NEXT:     "id": {{.+}},
// JSON-NEXT:     "kind": "CXXRecordDecl",
// JSON:          "isImplicit": true,
// JSON-NEXT:     "name": "Template2",
// JSON-NEXT:     "tagUsed": "struct"
// JSON-NEXT:   }
// JSON-NEXT: ]
} // namespace TestPartialSpecNTTP

namespace GH153540 {
// DUMP-LABEL: NamespaceDecl {{.*}} GH153540 external-linkage{{$}}
// JSON-LABEL: "name": "GH153540",

  namespace N {
    template<typename T> struct S { S(T); };
  }
  void f() {
    N::S(0);
  }

// DUMP:      FunctionDecl {{.*}} f 'void ()'
// DUMP-NEXT: CompoundStmt
// DUMP-NEXT: CXXFunctionalCastExpr {{.*}} 'N::S<int>':'GH153540::N::S<int>'
// DUMP-NEXT: CXXConstructExpr {{.*}} <col:5, col:11> 'N::S<int>':'GH153540::N::S<int>' 'void (int)'

// JSON:      "kind": "FunctionDecl",
// JSON:      "name": "f",
// JSON-NEXT: "mangledName": {{.*}},
// JSON-NEXT: "type": {
// JSON-NEXT:   "qualType": "void ()"
// JSON-NEXT: },
// JSON-NEXT: "inner": [
// JSON-NEXT:   {
// JSON-NEXT:     "id": {{.*}},
// JSON-NEXT:     "kind": "CompoundStmt",
// JSON:          "inner": [
// JSON-NEXT:       {
// JSON-NEXT:         "id": {{.*}},
// JSON-NEXT:         "kind": "CXXFunctionalCastExpr",
// JSON:              "type": {
// JSON-NEXT:           "desugaredQualType": "GH153540::N::S<int>",
// JSON-NEXT:           "qualType": "N::S<int>"
// JSON-NEXT:         },
// JSON-NEXT:         "valueCategory": "prvalue",
// JSON-NEXT:         "castKind": "ConstructorConversion",
// JSON-NEXT:         "conversionFunc": {
// JSON-NEXT:           "id": {{.*}},
// JSON-NEXT:           "kind": "CXXConstructorDecl",
// JSON-NEXT:           "name": "S",
// JSON-NEXT:           "type": {
// JSON-NEXT:             "qualType": "void (int)"
// JSON-NEXT:           }
// JSON-NEXT:         },
// JSON-NEXT:         "inner": [
// JSON-NEXT:           {
// JSON-NEXT:             "id": {{.*}},
// JSON-NEXT:             "kind": "CXXConstructExpr",
// JSON-NEXT:             "range": {
// JSON-NEXT:               "begin": {
// JSON-NEXT:                 "offset": {{.*}},
// JSON-NEXT:                 "col": 5,
// JSON-NEXT:                 "tokLen": {{.*}}
// JSON-NEXT:               },
// JSON-NEXT:               "end": {
// JSON-NEXT:                 "offset": {{.*}},
// JSON-NEXT:                 "col": 11,
// JSON-NEXT:                 "tokLen": {{.*}}
// JSON-NEXT:               }
// JSON-NEXT:             },
// JSON-NEXT:             "type": {
// JSON-NEXT:               "desugaredQualType": "GH153540::N::S<int>",
// JSON-NEXT:               "qualType": "N::S<int>"
// JSON-NEXT:             },
// JSON-NEXT:             "valueCategory": "prvalue",
// JSON-NEXT:             "ctorType": {
// JSON-NEXT:               "qualType": "void (int)"
// JSON-NEXT:             },
} // namespace GH153540

namespace AliasDependentTemplateSpecializationType {
// DUMP-LABEL: NamespaceDecl {{.*}} AliasDependentTemplateSpecializationType external-linkage{{$}}
// JSON-LABEL: "name": "AliasDependentTemplateSpecializationType",

  template<template<class> class TT> using T1 = TT<int>;
  template<class T> using T2 = T1<T::template X>;

// DUMP:      TypeAliasDecl {{.*}} T2 'T1<T::template X>':'T::template X<int>'
// DUMP-NEXT: `-TemplateSpecializationType {{.*}} 'T1<T::template X>' sugar dependent alias
// DUMP-NEXT:   |-name: 'T1':'AliasDependentTemplateSpecializationType::T1' qualified
// DUMP-NEXT:   | `-TypeAliasTemplateDecl {{.*}} T1
// DUMP-NEXT:   |-TemplateArgument template 'T::template X':'type-parameter-0-0::template X' dependent
// DUMP-NEXT:   | `-NestedNameSpecifier TypeSpec 'T'
// DUMP-NEXT:   `-TemplateSpecializationType {{.*}} 'T::template X<int>' dependent
// DUMP-NEXT:     |-name: 'T::template X':'type-parameter-0-0::template X' subst index 0 final
// DUMP-NEXT:     | |-parameter: TemplateTemplateParmDecl {{.*}} depth 0 index 0 TT
// DUMP-NEXT:     | |-associated TypeAliasTemplate {{.*}} 'T1'
// DUMP-NEXT:     | `-replacement: 'T::template X':'type-parameter-0-0::template X' dependent
// DUMP-NEXT:     |   `-NestedNameSpecifier TypeSpec 'T'
// DUMP-NEXT:     `-TemplateArgument type 'int'
// DUMP-NEXT:       `-BuiltinType {{.*}} 'int'

// FIXME: a non-trivial amount of data present in textual dump is nowhere
//        to be found in JSON dump, because JSON dumper cannot dump the data
//        inside TemplateNames.

// JSON:      "kind": "TypeAliasTemplateDecl"
// JSON:      "name": "T2",
// JSON:       "kind": "TypeAliasDecl",
// JSON:       "name": "T2",
// JSON-NEXT:  "type": {
// JSON-NEXT:    "desugaredQualType": "T::template X<int>",
// JSON-NEXT:    "qualType": "T1<T::template X>"
// JSON-NEXT:  },
// JSON-NEXT:  "inner": [
// JSON-NEXT:    {
// JSON-NEXT:      "id": {{.*}},
// JSON-NEXT:      "kind": "TemplateSpecializationType",
// JSON-NEXT:      "type": {
// JSON-NEXT:        "qualType": "T1<T::template X>"
// JSON-NEXT:      },
// JSON-NEXT:      "isDependent": true,
// JSON-NEXT:      "isInstantiationDependent": true,
// JSON-NEXT:      "isAlias": true,
// JSON-NEXT:      "templateName": "T1",
// JSON-NEXT:      "inner": [
// JSON-NEXT:        {
// JSON-NEXT:          "kind": "TemplateArgument"
// JSON-NEXT:        },
// JSON-NEXT:        {
// JSON-NEXT:          "id": {{.*}},
// JSON-NEXT:          "kind": "TemplateSpecializationType",
// JSON-NEXT:          "type": {
// JSON-NEXT:            "qualType": "T::template X<int>"
// JSON-NEXT:          },
// JSON-NEXT:          "isDependent": true,
// JSON-NEXT:          "isInstantiationDependent": true,
// JSON-NEXT:          "templateName": "T::template X",
// JSON-NEXT:          "inner": [
// JSON-NEXT:            {
// JSON-NEXT:              "kind": "TemplateArgument",
// JSON-NEXT:              "type": {
// JSON-NEXT:                "qualType": "int"
// JSON-NEXT:              },
// JSON-NEXT:              "inner": [
// JSON-NEXT:                {
// JSON-NEXT:                  "id": {{.*}},
// JSON-NEXT:                  "kind": "BuiltinType",
// JSON-NEXT:                  "type": {
// JSON-NEXT:                    "qualType": "int"
// JSON-NEXT:                  }
} // namespace

namespace TestAbbreviatedTemplateDecls {
// DUMP-LABEL: NamespaceDecl {{.*}} TestAbbreviatedTemplateDecls external-linkage{{$}}
// JSON-LABEL: "name": "TestAbbreviatedTemplateDecls",

  void abbreviated(auto);
  template<class T>
  void mixed(T, auto);

// DUMP: FunctionTemplateDecl {{.*}} <line:[[@LINE-4]]:3, col:24> col:8 abbreviated
// DUMP: FunctionTemplateDecl {{.*}} <line:[[@LINE-4]]:3, line:[[@LINE-3]]:21> col:8 mixed

// JSON:       "kind": "FunctionTemplateDecl",
// JSON-NEXT:  "loc": {
// JSON-NEXT:    "offset": {{.*}},
// JSON-NEXT:    "line": [[#@LINE-10]],
// JSON-NEXT:    "col": 8,
// JSON-NEXT:    "tokLen": {{.*}}
// JSON-NEXT:  },
// JSON-NEXT:  "range": {
// JSON-NEXT:    "begin": {
// JSON-NEXT:      "offset": {{.*}},
// JSON-NEXT:      "col": 3,
// JSON-NEXT:      "tokLen": {{.*}}
// JSON-NEXT:    },
// JSON-NEXT:    "end": {
// JSON-NEXT:      "offset": {{.*}},
// JSON-NEXT:      "col": 24,
// JSON-NEXT:      "tokLen": {{.*}}
// JSON-NEXT:    }
// JSON-NEXT:  },
// JSON-NEXT:  "name": "abbreviated",

// JSON:       "kind": "FunctionTemplateDecl",
// JSON-NEXT:  "loc": {
// JSON-NEXT:    "offset": {{.*}},
// JSON-NEXT:    "line": {{.*}},
// JSON-NEXT:    "col": 8,
// JSON-NEXT:    "tokLen": {{.*}}
// JSON-NEXT:  },
// JSON-NEXT:  "range": {
// JSON-NEXT:    "begin": {
// JSON-NEXT:      "offset": {{.*}},
// JSON-NEXT:      "line": [[#@LINE-37]],
// JSON-NEXT:      "col": 3,
// JSON-NEXT:      "tokLen": {{.*}}
// JSON-NEXT:    },
// JSON-NEXT:    "end": {
// JSON-NEXT:      "offset": {{.*}},
// JSON-NEXT:      "line": [[#@LINE-42]],
// JSON-NEXT:      "col": 21,
// JSON-NEXT:      "tokLen": {{.*}}
// JSON-NEXT:    }
// JSON-NEXT:  },
// JSON-NEXT:  "name": "mixed",

} // namespace TestAbbreviatedTemplateDecls
