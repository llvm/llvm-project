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
}
}

namespace test7 {
  template <template<class> class TT> struct A {};
  template <class...> class B {};
  template struct A<B>;
// DUMP-LABEL: NamespaceDecl {{.*}} test7{{$}}
// DUMP:       ClassTemplateSpecializationDecl {{.*}} struct A definition explicit_instantiation_definition strict-pack-match{{$}}
} // namespce test7

namespace test8 {
template<_Complex int x>
struct pr126341;
template<>
struct pr126341<{1, 2}>;
// DUMP-LABEL: NamespaceDecl {{.*}} test8{{$}}
// DUMP-NEXT:  |-ClassTemplateDecl {{.*}} pr126341
// DUMP:       `-ClassTemplateSpecializationDecl {{.*}} pr126341
// DUMP:         `-TemplateArgument structural value '1+2i'
} // namespace test8

namespace TestMemberPointerPartialSpec {
  template <class> struct A;
  template <class T1, class T2> struct A<T1 T2::*>;
// DUMP-LABEL: NamespaceDecl {{.+}} TestMemberPointerPartialSpec{{$}}
// DUMP:       ClassTemplatePartialSpecializationDecl {{.*}} struct A
// DUMP-NEXT:  |-TemplateArgument type 'type-parameter-0-0 type-parameter-0-1::*'
// DUMP-NEXT:  | `-MemberPointerType {{.+}} 'type-parameter-0-0 type-parameter-0-1::*' dependent
// DUMP-NEXT:  |   |-TemplateTypeParmType {{.+}} 'type-parameter-0-1' dependent depth 0 index 1
// DUMP-NEXT:  |   `-TemplateTypeParmType {{.+}} 'type-parameter-0-0' dependent depth 0 index 0
} // namespace TestMemberPointerPartialSpec

namespace TestDependentMemberPointer {
  template <class U> struct A {
    using X = int U::*;
    using Y = int U::test::*;
    using Z = int U::template V<int>::*;
  };
// DUMP-LABEL: NamespaceDecl {{.+}} TestDependentMemberPointer{{$}}
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
// DUMP-NEXT:      |-DependentTemplateSpecializationType {{.+}} 'U::template V<int>' dependent
// DUMP-NEXT:      `-BuiltinType {{.+}} 'int'
} // namespace TestDependentMemberPointer

namespace TestPartialSpecNTTP {
// DUMP-LABEL: NamespaceDecl {{.+}} TestPartialSpecNTTP{{$}}
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

  template <typename U1, bool U3, bool U2>
  struct Template2<Template1<U1, U2>, U3> {};
// DUMP:      ClassTemplatePartialSpecializationDecl {{.+}} struct Template2 definition explicit_specialization
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
} // namespace TestPartialSpecNTTP

namespace GH153540 {
// DUMP-LABEL: NamespaceDecl {{.*}} GH153540{{$}}

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
} // namespace GH153540

// NOTE: CHECK lines have been autogenerated by gen_ast_dump_json_test.py


// JSON-NOT: {{^}}Dumping
// JSON:  "kind": "TranslationUnitDecl",
// JSON-NEXT:  "loc": {},
// JSON-NEXT:  "range": {
// JSON-NEXT:   "begin": {},
// JSON-NEXT:   "end": {}
// JSON-NEXT:  },
// JSON-NEXT:  "inner": [
// JSON-NEXT:   {
// JSON-NEXT:    "id": "0x{{.*}}",
// JSON-NEXT:    "kind": "TypedefDecl",
// JSON-NEXT:    "loc": {},
// JSON-NEXT:    "range": {
// JSON-NEXT:     "begin": {},
// JSON-NEXT:     "end": {}
// JSON-NEXT:    },
// JSON-NEXT:    "isImplicit": true,
// JSON-NEXT:    "name": "__int128_t",
// JSON-NEXT:    "type": {
// JSON-NEXT:     "qualType": "__int128"
// JSON-NEXT:    },
// JSON-NEXT:    "inner": [
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "BuiltinType",
// JSON-NEXT:      "type": {
// JSON-NEXT:       "qualType": "__int128"
// JSON-NEXT:      }
// JSON-NEXT:     }
// JSON-NEXT:    ]
// JSON-NEXT:   },
// JSON-NEXT:   {
// JSON-NEXT:    "id": "0x{{.*}}",
// JSON-NEXT:    "kind": "TypedefDecl",
// JSON-NEXT:    "loc": {},
// JSON-NEXT:    "range": {
// JSON-NEXT:     "begin": {},
// JSON-NEXT:     "end": {}
// JSON-NEXT:    },
// JSON-NEXT:    "isImplicit": true,
// JSON-NEXT:    "name": "__uint128_t",
// JSON-NEXT:    "type": {
// JSON-NEXT:     "qualType": "unsigned __int128"
// JSON-NEXT:    },
// JSON-NEXT:    "inner": [
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "BuiltinType",
// JSON-NEXT:      "type": {
// JSON-NEXT:       "qualType": "unsigned __int128"
// JSON-NEXT:      }
// JSON-NEXT:     }
// JSON-NEXT:    ]
// JSON-NEXT:   },
// JSON-NEXT:   {
// JSON-NEXT:    "id": "0x{{.*}}",
// JSON-NEXT:    "kind": "TypedefDecl",
// JSON-NEXT:    "loc": {},
// JSON-NEXT:    "range": {
// JSON-NEXT:     "begin": {},
// JSON-NEXT:     "end": {}
// JSON-NEXT:    },
// JSON-NEXT:    "isImplicit": true,
// JSON-NEXT:    "name": "__NSConstantString",
// JSON-NEXT:    "type": {
// JSON-NEXT:     "qualType": "__NSConstantString_tag"
// JSON-NEXT:    },
// JSON-NEXT:    "inner": [
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "RecordType",
// JSON-NEXT:      "type": {
// JSON-NEXT:       "qualType": "__NSConstantString_tag"
// JSON-NEXT:      },
// JSON-NEXT:      "decl": {
// JSON-NEXT:       "id": "0x{{.*}}",
// JSON-NEXT:       "kind": "CXXRecordDecl",
// JSON-NEXT:       "name": "__NSConstantString_tag"
// JSON-NEXT:      }
// JSON-NEXT:     }
// JSON-NEXT:    ]
// JSON-NEXT:   },
// JSON-NEXT:   {
// JSON-NEXT:    "id": "0x{{.*}}",
// JSON-NEXT:    "kind": "TypedefDecl",
// JSON-NEXT:    "loc": {},
// JSON-NEXT:    "range": {
// JSON-NEXT:     "begin": {},
// JSON-NEXT:     "end": {}
// JSON-NEXT:    },
// JSON-NEXT:    "isImplicit": true,
// JSON-NEXT:    "name": "__builtin_ms_va_list",
// JSON-NEXT:    "type": {
// JSON-NEXT:     "qualType": "char *"
// JSON-NEXT:    },
// JSON-NEXT:    "inner": [
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "PointerType",
// JSON-NEXT:      "type": {
// JSON-NEXT:       "qualType": "char *"
// JSON-NEXT:      },
// JSON-NEXT:      "inner": [
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "BuiltinType",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "char"
// JSON-NEXT:        }
// JSON-NEXT:       }
// JSON-NEXT:      ]
// JSON-NEXT:     }
// JSON-NEXT:    ]
// JSON-NEXT:   },
// JSON-NEXT:   {
// JSON-NEXT:    "id": "0x{{.*}}",
// JSON-NEXT:    "kind": "TypedefDecl",
// JSON-NEXT:    "loc": {},
// JSON-NEXT:    "range": {
// JSON-NEXT:     "begin": {},
// JSON-NEXT:     "end": {}
// JSON-NEXT:    },
// JSON-NEXT:    "isImplicit": true,
// JSON-NEXT:    "name": "__builtin_va_list",
// JSON-NEXT:    "type": {
// JSON-NEXT:     "qualType": "__va_list_tag[1]"
// JSON-NEXT:    },
// JSON-NEXT:    "inner": [
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "ConstantArrayType",
// JSON-NEXT:      "type": {
// JSON-NEXT:       "qualType": "__va_list_tag[1]"
// JSON-NEXT:      },
// JSON-NEXT:      "size": 1,
// JSON-NEXT:      "inner": [
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "RecordType",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "__va_list_tag"
// JSON-NEXT:        },
// JSON-NEXT:        "decl": {
// JSON-NEXT:         "id": "0x{{.*}}",
// JSON-NEXT:         "kind": "CXXRecordDecl",
// JSON-NEXT:         "name": "__va_list_tag"
// JSON-NEXT:        }
// JSON-NEXT:       }
// JSON-NEXT:      ]
// JSON-NEXT:     }
// JSON-NEXT:    ]
// JSON-NEXT:   },
// JSON-NEXT:   {
// JSON-NEXT:    "id": "0x{{.*}}",
// JSON-NEXT:    "kind": "ClassTemplateDecl",
// JSON-NEXT:    "loc": {
// JSON-NEXT:     "offset": 812,
// JSON-NEXT:     "file": "{{.*}}",
// JSON-NEXT:     "line": 15,
// JSON-NEXT:     "col": 8,
// JSON-NEXT:     "tokLen": 3
// JSON-NEXT:    },
// JSON-NEXT:    "range": {
// JSON-NEXT:     "begin": {
// JSON-NEXT:      "offset": 765,
// JSON-NEXT:      "line": 14,
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 8
// JSON-NEXT:     },
// JSON-NEXT:     "end": {
// JSON-NEXT:      "offset": 879,
// JSON-NEXT:      "line": 19,
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 1
// JSON-NEXT:     }
// JSON-NEXT:    },
// JSON-NEXT:    "name": "foo",
// JSON-NEXT:    "inner": [
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "NonTypeTemplateParmDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": 779,
// JSON-NEXT:       "line": 14,
// JSON-NEXT:       "col": 15,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 775,
// JSON-NEXT:        "col": 11,
// JSON-NEXT:        "tokLen": 3
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 779,
// JSON-NEXT:        "col": 15,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "isReferenced": true,
// JSON-NEXT:      "name": "X",
// JSON-NEXT:      "type": {
// JSON-NEXT:       "qualType": "int"
// JSON-NEXT:      },
// JSON-NEXT:      "depth": 0,
// JSON-NEXT:      "index": 0
// JSON-NEXT:     },
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "TemplateTypeParmDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": 791,
// JSON-NEXT:       "col": 27,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 782,
// JSON-NEXT:        "col": 18,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 791,
// JSON-NEXT:        "col": 27,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "isReferenced": true,
// JSON-NEXT:      "name": "Y",
// JSON-NEXT:      "tagUsed": "typename",
// JSON-NEXT:      "depth": 0,
// JSON-NEXT:      "index": 1
// JSON-NEXT:     },
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "NonTypeTemplateParmDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": 798,
// JSON-NEXT:       "col": 34,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 794,
// JSON-NEXT:        "col": 30,
// JSON-NEXT:        "tokLen": 3
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 802,
// JSON-NEXT:        "col": 38,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "isReferenced": true,
// JSON-NEXT:      "name": "Z",
// JSON-NEXT:      "type": {
// JSON-NEXT:       "qualType": "int"
// JSON-NEXT:      },
// JSON-NEXT:      "depth": 0,
// JSON-NEXT:      "index": 2,
// JSON-NEXT:      "defaultArg": {
// JSON-NEXT:       "kind": "TemplateArgument",
// JSON-NEXT:       "isExpr": true
// JSON-NEXT:      },
// JSON-NEXT:      "inner": [
// JSON-NEXT:       {
// JSON-NEXT:        "kind": "TemplateArgument",
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 802,
// JSON-NEXT:          "col": 38,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 802,
// JSON-NEXT:          "col": 38,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "isExpr": true,
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "IntegerLiteral",
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 802,
// JSON-NEXT:            "col": 38,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 802,
// JSON-NEXT:            "col": 38,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "int"
// JSON-NEXT:          },
// JSON-NEXT:          "valueCategory": "prvalue",
// JSON-NEXT:          "value": "5"
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       }
// JSON-NEXT:      ]
// JSON-NEXT:     },
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "CXXRecordDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": 812,
// JSON-NEXT:       "line": 15,
// JSON-NEXT:       "col": 8,
// JSON-NEXT:       "tokLen": 3
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 805,
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 6
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 879,
// JSON-NEXT:        "line": 19,
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "name": "foo",
// JSON-NEXT:      "tagUsed": "struct",
// JSON-NEXT:      "completeDefinition": true,
// JSON-NEXT:      "definitionData": {
// JSON-NEXT:       "canConstDefaultInit": true,
// JSON-NEXT:       "copyAssign": {
// JSON-NEXT:        "hasConstParam": true,
// JSON-NEXT:        "implicitHasConstParam": true,
// JSON-NEXT:        "needsImplicit": true,
// JSON-NEXT:        "simple": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       },
// JSON-NEXT:       "copyCtor": {
// JSON-NEXT:        "hasConstParam": true,
// JSON-NEXT:        "implicitHasConstParam": true,
// JSON-NEXT:        "needsImplicit": true,
// JSON-NEXT:        "simple": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       },
// JSON-NEXT:       "defaultCtor": {
// JSON-NEXT:        "defaultedIsConstexpr": true,
// JSON-NEXT:        "exists": true,
// JSON-NEXT:        "nonTrivial": true,
// JSON-NEXT:        "userProvided": true
// JSON-NEXT:       },
// JSON-NEXT:       "dtor": {
// JSON-NEXT:        "irrelevant": true,
// JSON-NEXT:        "needsImplicit": true,
// JSON-NEXT:        "simple": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       },
// JSON-NEXT:       "hasUserDeclaredConstructor": true,
// JSON-NEXT:       "isStandardLayout": true,
// JSON-NEXT:       "isTriviallyCopyable": true,
// JSON-NEXT:       "moveAssign": {
// JSON-NEXT:        "exists": true,
// JSON-NEXT:        "needsImplicit": true,
// JSON-NEXT:        "simple": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       },
// JSON-NEXT:       "moveCtor": {
// JSON-NEXT:        "exists": true,
// JSON-NEXT:        "needsImplicit": true,
// JSON-NEXT:        "simple": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "inner": [
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "CXXRecordDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 812,
// JSON-NEXT:         "line": 15,
// JSON-NEXT:         "col": 8,
// JSON-NEXT:         "tokLen": 3
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 805,
// JSON-NEXT:          "col": 1,
// JSON-NEXT:          "tokLen": 6
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 812,
// JSON-NEXT:          "col": 8,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "isImplicit": true,
// JSON-NEXT:        "name": "foo",
// JSON-NEXT:        "tagUsed": "struct"
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "FieldDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 824,
// JSON-NEXT:         "line": 16,
// JSON-NEXT:         "col": 7,
// JSON-NEXT:         "tokLen": 8
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 820,
// JSON-NEXT:          "col": 3,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 824,
// JSON-NEXT:          "col": 7,
// JSON-NEXT:          "tokLen": 8
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "name": "constant",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "int"
// JSON-NEXT:        }
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "CXXConstructorDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 836,
// JSON-NEXT:         "line": 17,
// JSON-NEXT:         "col": 3,
// JSON-NEXT:         "tokLen": 3
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 836,
// JSON-NEXT:          "col": 3,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 843,
// JSON-NEXT:          "col": 10,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "name": "foo<X, Y, Z>",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "void ()"
// JSON-NEXT:        },
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "CompoundStmt",
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 842,
// JSON-NEXT:            "col": 9,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 843,
// JSON-NEXT:            "col": 10,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          }
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "CXXMethodDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 849,
// JSON-NEXT:         "line": 18,
// JSON-NEXT:         "col": 5,
// JSON-NEXT:         "tokLen": 6
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 847,
// JSON-NEXT:          "col": 3,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 877,
// JSON-NEXT:          "col": 33,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "name": "getSum",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "Y ()"
// JSON-NEXT:        },
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "CompoundStmt",
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 858,
// JSON-NEXT:            "col": 14,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 877,
// JSON-NEXT:            "col": 33,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "inner": [
// JSON-NEXT:           {
// JSON-NEXT:            "id": "0x{{.*}}",
// JSON-NEXT:            "kind": "ReturnStmt",
// JSON-NEXT:            "range": {
// JSON-NEXT:             "begin": {
// JSON-NEXT:              "offset": 860,
// JSON-NEXT:              "col": 16,
// JSON-NEXT:              "tokLen": 6
// JSON-NEXT:             },
// JSON-NEXT:             "end": {
// JSON-NEXT:              "offset": 874,
// JSON-NEXT:              "col": 30,
// JSON-NEXT:              "tokLen": 1
// JSON-NEXT:             }
// JSON-NEXT:            },
// JSON-NEXT:            "inner": [
// JSON-NEXT:             {
// JSON-NEXT:              "id": "0x{{.*}}",
// JSON-NEXT:              "kind": "CXXUnresolvedConstructExpr",
// JSON-NEXT:              "range": {
// JSON-NEXT:               "begin": {
// JSON-NEXT:                "offset": 867,
// JSON-NEXT:                "col": 23,
// JSON-NEXT:                "tokLen": 1
// JSON-NEXT:               },
// JSON-NEXT:               "end": {
// JSON-NEXT:                "offset": 874,
// JSON-NEXT:                "col": 30,
// JSON-NEXT:                "tokLen": 1
// JSON-NEXT:               }
// JSON-NEXT:              },
// JSON-NEXT:              "type": {
// JSON-NEXT:               "qualType": "Y"
// JSON-NEXT:              },
// JSON-NEXT:              "valueCategory": "prvalue",
// JSON-NEXT:              "inner": [
// JSON-NEXT:               {
// JSON-NEXT:                "id": "0x{{.*}}",
// JSON-NEXT:                "kind": "BinaryOperator",
// JSON-NEXT:                "range": {
// JSON-NEXT:                 "begin": {
// JSON-NEXT:                  "offset": 869,
// JSON-NEXT:                  "col": 25,
// JSON-NEXT:                  "tokLen": 1
// JSON-NEXT:                 },
// JSON-NEXT:                 "end": {
// JSON-NEXT:                  "offset": 873,
// JSON-NEXT:                  "col": 29,
// JSON-NEXT:                  "tokLen": 1
// JSON-NEXT:                 }
// JSON-NEXT:                },
// JSON-NEXT:                "type": {
// JSON-NEXT:                 "qualType": "int"
// JSON-NEXT:                },
// JSON-NEXT:                "valueCategory": "prvalue",
// JSON-NEXT:                "opcode": "+",
// JSON-NEXT:                "inner": [
// JSON-NEXT:                 {
// JSON-NEXT:                  "id": "0x{{.*}}",
// JSON-NEXT:                  "kind": "DeclRefExpr",
// JSON-NEXT:                  "range": {
// JSON-NEXT:                   "begin": {
// JSON-NEXT:                    "offset": 869,
// JSON-NEXT:                    "col": 25,
// JSON-NEXT:                    "tokLen": 1
// JSON-NEXT:                   },
// JSON-NEXT:                   "end": {
// JSON-NEXT:                    "offset": 869,
// JSON-NEXT:                    "col": 25,
// JSON-NEXT:                    "tokLen": 1
// JSON-NEXT:                   }
// JSON-NEXT:                  },
// JSON-NEXT:                  "type": {
// JSON-NEXT:                   "qualType": "int"
// JSON-NEXT:                  },
// JSON-NEXT:                  "valueCategory": "prvalue",
// JSON-NEXT:                  "referencedDecl": {
// JSON-NEXT:                   "id": "0x{{.*}}",
// JSON-NEXT:                   "kind": "NonTypeTemplateParmDecl",
// JSON-NEXT:                   "name": "X",
// JSON-NEXT:                   "type": {
// JSON-NEXT:                    "qualType": "int"
// JSON-NEXT:                   }
// JSON-NEXT:                  }
// JSON-NEXT:                 },
// JSON-NEXT:                 {
// JSON-NEXT:                  "id": "0x{{.*}}",
// JSON-NEXT:                  "kind": "DeclRefExpr",
// JSON-NEXT:                  "range": {
// JSON-NEXT:                   "begin": {
// JSON-NEXT:                    "offset": 873,
// JSON-NEXT:                    "col": 29,
// JSON-NEXT:                    "tokLen": 1
// JSON-NEXT:                   },
// JSON-NEXT:                   "end": {
// JSON-NEXT:                    "offset": 873,
// JSON-NEXT:                    "col": 29,
// JSON-NEXT:                    "tokLen": 1
// JSON-NEXT:                   }
// JSON-NEXT:                  },
// JSON-NEXT:                  "type": {
// JSON-NEXT:                   "qualType": "int"
// JSON-NEXT:                  },
// JSON-NEXT:                  "valueCategory": "prvalue",
// JSON-NEXT:                  "referencedDecl": {
// JSON-NEXT:                   "id": "0x{{.*}}",
// JSON-NEXT:                   "kind": "NonTypeTemplateParmDecl",
// JSON-NEXT:                   "name": "Z",
// JSON-NEXT:                   "type": {
// JSON-NEXT:                    "qualType": "int"
// JSON-NEXT:                   }
// JSON-NEXT:                  }
// JSON-NEXT:                 }
// JSON-NEXT:                ]
// JSON-NEXT:               }
// JSON-NEXT:              ]
// JSON-NEXT:             }
// JSON-NEXT:            ]
// JSON-NEXT:           }
// JSON-NEXT:          ]
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       }
// JSON-NEXT:      ]
// JSON-NEXT:     },
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "ClassTemplateSpecializationDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": 812,
// JSON-NEXT:       "line": 15,
// JSON-NEXT:       "col": 8,
// JSON-NEXT:       "tokLen": 3
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 765,
// JSON-NEXT:        "line": 14,
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 879,
// JSON-NEXT:        "line": 19,
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "name": "foo",
// JSON-NEXT:      "tagUsed": "struct",
// JSON-NEXT:      "completeDefinition": true,
// JSON-NEXT:      "definitionData": {
// JSON-NEXT:       "canConstDefaultInit": true,
// JSON-NEXT:       "canPassInRegisters": true,
// JSON-NEXT:       "copyAssign": {
// JSON-NEXT:        "hasConstParam": true,
// JSON-NEXT:        "implicitHasConstParam": true,
// JSON-NEXT:        "needsImplicit": true,
// JSON-NEXT:        "simple": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       },
// JSON-NEXT:       "copyCtor": {
// JSON-NEXT:        "hasConstParam": true,
// JSON-NEXT:        "implicitHasConstParam": true,
// JSON-NEXT:        "simple": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       },
// JSON-NEXT:       "defaultCtor": {
// JSON-NEXT:        "defaultedIsConstexpr": true,
// JSON-NEXT:        "exists": true,
// JSON-NEXT:        "nonTrivial": true,
// JSON-NEXT:        "userProvided": true
// JSON-NEXT:       },
// JSON-NEXT:       "dtor": {
// JSON-NEXT:        "irrelevant": true,
// JSON-NEXT:        "simple": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       },
// JSON-NEXT:       "hasUserDeclaredConstructor": true,
// JSON-NEXT:       "isStandardLayout": true,
// JSON-NEXT:       "isTriviallyCopyable": true,
// JSON-NEXT:       "moveAssign": {
// JSON-NEXT:        "exists": true,
// JSON-NEXT:        "needsImplicit": true,
// JSON-NEXT:        "simple": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       },
// JSON-NEXT:       "moveCtor": {
// JSON-NEXT:        "exists": true,
// JSON-NEXT:        "simple": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "inner": [
// JSON-NEXT:       {
// JSON-NEXT:        "kind": "TemplateArgument",
// JSON-NEXT:        "value": 5
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "kind": "TemplateArgument",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "int"
// JSON-NEXT:        },
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "BuiltinType",
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "int"
// JSON-NEXT:          }
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "kind": "TemplateArgument",
// JSON-NEXT:        "value": 5
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "CXXRecordDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 812,
// JSON-NEXT:         "line": 15,
// JSON-NEXT:         "col": 8,
// JSON-NEXT:         "tokLen": 3
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 805,
// JSON-NEXT:          "col": 1,
// JSON-NEXT:          "tokLen": 6
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 812,
// JSON-NEXT:          "col": 8,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "isImplicit": true,
// JSON-NEXT:        "name": "foo",
// JSON-NEXT:        "tagUsed": "struct"
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "FieldDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 824,
// JSON-NEXT:         "line": 16,
// JSON-NEXT:         "col": 7,
// JSON-NEXT:         "tokLen": 8
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 820,
// JSON-NEXT:          "col": 3,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 824,
// JSON-NEXT:          "col": 7,
// JSON-NEXT:          "tokLen": 8
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "name": "constant",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "int"
// JSON-NEXT:        }
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "CXXConstructorDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 836,
// JSON-NEXT:         "line": 17,
// JSON-NEXT:         "col": 3,
// JSON-NEXT:         "tokLen": 3
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 836,
// JSON-NEXT:          "col": 3,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 843,
// JSON-NEXT:          "col": 10,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "isUsed": true,
// JSON-NEXT:        "name": "foo",
// JSON-NEXT:        "mangledName": "_ZN3fooILi5EiLi5EEC1Ev",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "void ()"
// JSON-NEXT:        },
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "CompoundStmt",
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 842,
// JSON-NEXT:            "col": 9,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 843,
// JSON-NEXT:            "col": 10,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          }
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "CXXMethodDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 849,
// JSON-NEXT:         "line": 18,
// JSON-NEXT:         "col": 5,
// JSON-NEXT:         "tokLen": 6
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 847,
// JSON-NEXT:          "col": 3,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 877,
// JSON-NEXT:          "col": 33,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "isUsed": true,
// JSON-NEXT:        "name": "getSum",
// JSON-NEXT:        "mangledName": "_ZN3fooILi5EiLi5EE6getSumEv",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "int ()"
// JSON-NEXT:        },
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "CompoundStmt",
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 858,
// JSON-NEXT:            "col": 14,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 877,
// JSON-NEXT:            "col": 33,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "inner": [
// JSON-NEXT:           {
// JSON-NEXT:            "id": "0x{{.*}}",
// JSON-NEXT:            "kind": "ReturnStmt",
// JSON-NEXT:            "range": {
// JSON-NEXT:             "begin": {
// JSON-NEXT:              "offset": 860,
// JSON-NEXT:              "col": 16,
// JSON-NEXT:              "tokLen": 6
// JSON-NEXT:             },
// JSON-NEXT:             "end": {
// JSON-NEXT:              "offset": 874,
// JSON-NEXT:              "col": 30,
// JSON-NEXT:              "tokLen": 1
// JSON-NEXT:             }
// JSON-NEXT:            },
// JSON-NEXT:            "inner": [
// JSON-NEXT:             {
// JSON-NEXT:              "id": "0x{{.*}}",
// JSON-NEXT:              "kind": "CXXFunctionalCastExpr",
// JSON-NEXT:              "range": {
// JSON-NEXT:               "begin": {
// JSON-NEXT:                "offset": 867,
// JSON-NEXT:                "col": 23,
// JSON-NEXT:                "tokLen": 1
// JSON-NEXT:               },
// JSON-NEXT:               "end": {
// JSON-NEXT:                "offset": 874,
// JSON-NEXT:                "col": 30,
// JSON-NEXT:                "tokLen": 1
// JSON-NEXT:               }
// JSON-NEXT:              },
// JSON-NEXT:              "type": {
// JSON-NEXT:               "qualType": "int"
// JSON-NEXT:              },
// JSON-NEXT:              "valueCategory": "prvalue",
// JSON-NEXT:              "castKind": "NoOp",
// JSON-NEXT:              "inner": [
// JSON-NEXT:               {
// JSON-NEXT:                "id": "0x{{.*}}",
// JSON-NEXT:                "kind": "BinaryOperator",
// JSON-NEXT:                "range": {
// JSON-NEXT:                 "begin": {
// JSON-NEXT:                  "offset": 869,
// JSON-NEXT:                  "col": 25,
// JSON-NEXT:                  "tokLen": 1
// JSON-NEXT:                 },
// JSON-NEXT:                 "end": {
// JSON-NEXT:                  "offset": 873,
// JSON-NEXT:                  "col": 29,
// JSON-NEXT:                  "tokLen": 1
// JSON-NEXT:                 }
// JSON-NEXT:                },
// JSON-NEXT:                "type": {
// JSON-NEXT:                 "qualType": "int"
// JSON-NEXT:                },
// JSON-NEXT:                "valueCategory": "prvalue",
// JSON-NEXT:                "opcode": "+",
// JSON-NEXT:                "inner": [
// JSON-NEXT:                 {
// JSON-NEXT:                  "id": "0x{{.*}}",
// JSON-NEXT:                  "kind": "SubstNonTypeTemplateParmExpr",
// JSON-NEXT:                  "range": {
// JSON-NEXT:                   "begin": {
// JSON-NEXT:                    "offset": 869,
// JSON-NEXT:                    "col": 25,
// JSON-NEXT:                    "tokLen": 1
// JSON-NEXT:                   },
// JSON-NEXT:                   "end": {
// JSON-NEXT:                    "offset": 869,
// JSON-NEXT:                    "col": 25,
// JSON-NEXT:                    "tokLen": 1
// JSON-NEXT:                   }
// JSON-NEXT:                  },
// JSON-NEXT:                  "type": {
// JSON-NEXT:                   "qualType": "int"
// JSON-NEXT:                  },
// JSON-NEXT:                  "valueCategory": "prvalue",
// JSON-NEXT:                  "inner": [
// JSON-NEXT:                   {
// JSON-NEXT:                    "id": "0x{{.*}}",
// JSON-NEXT:                    "kind": "NonTypeTemplateParmDecl",
// JSON-NEXT:                    "loc": {
// JSON-NEXT:                     "offset": 779,
// JSON-NEXT:                     "line": 14,
// JSON-NEXT:                     "col": 15,
// JSON-NEXT:                     "tokLen": 1
// JSON-NEXT:                    },
// JSON-NEXT:                    "range": {
// JSON-NEXT:                     "begin": {
// JSON-NEXT:                      "offset": 775,
// JSON-NEXT:                      "col": 11,
// JSON-NEXT:                      "tokLen": 3
// JSON-NEXT:                     },
// JSON-NEXT:                     "end": {
// JSON-NEXT:                      "offset": 779,
// JSON-NEXT:                      "col": 15,
// JSON-NEXT:                      "tokLen": 1
// JSON-NEXT:                     }
// JSON-NEXT:                    },
// JSON-NEXT:                    "isReferenced": true,
// JSON-NEXT:                    "name": "X",
// JSON-NEXT:                    "type": {
// JSON-NEXT:                     "qualType": "int"
// JSON-NEXT:                    },
// JSON-NEXT:                    "depth": 0,
// JSON-NEXT:                    "index": 0
// JSON-NEXT:                   },
// JSON-NEXT:                   {
// JSON-NEXT:                    "id": "0x{{.*}}",
// JSON-NEXT:                    "kind": "IntegerLiteral",
// JSON-NEXT:                    "range": {
// JSON-NEXT:                     "begin": {
// JSON-NEXT:                      "offset": 869,
// JSON-NEXT:                      "line": 18,
// JSON-NEXT:                      "col": 25,
// JSON-NEXT:                      "tokLen": 1
// JSON-NEXT:                     },
// JSON-NEXT:                     "end": {
// JSON-NEXT:                      "offset": 869,
// JSON-NEXT:                      "col": 25,
// JSON-NEXT:                      "tokLen": 1
// JSON-NEXT:                     }
// JSON-NEXT:                    },
// JSON-NEXT:                    "type": {
// JSON-NEXT:                     "qualType": "int"
// JSON-NEXT:                    },
// JSON-NEXT:                    "valueCategory": "prvalue",
// JSON-NEXT:                    "value": "5"
// JSON-NEXT:                   }
// JSON-NEXT:                  ]
// JSON-NEXT:                 },
// JSON-NEXT:                 {
// JSON-NEXT:                  "id": "0x{{.*}}",
// JSON-NEXT:                  "kind": "SubstNonTypeTemplateParmExpr",
// JSON-NEXT:                  "range": {
// JSON-NEXT:                   "begin": {
// JSON-NEXT:                    "offset": 873,
// JSON-NEXT:                    "col": 29,
// JSON-NEXT:                    "tokLen": 1
// JSON-NEXT:                   },
// JSON-NEXT:                   "end": {
// JSON-NEXT:                    "offset": 873,
// JSON-NEXT:                    "col": 29,
// JSON-NEXT:                    "tokLen": 1
// JSON-NEXT:                   }
// JSON-NEXT:                  },
// JSON-NEXT:                  "type": {
// JSON-NEXT:                   "qualType": "int"
// JSON-NEXT:                  },
// JSON-NEXT:                  "valueCategory": "prvalue",
// JSON-NEXT:                  "inner": [
// JSON-NEXT:                   {
// JSON-NEXT:                    "id": "0x{{.*}}",
// JSON-NEXT:                    "kind": "NonTypeTemplateParmDecl",
// JSON-NEXT:                    "loc": {
// JSON-NEXT:                     "offset": 798,
// JSON-NEXT:                     "line": 14,
// JSON-NEXT:                     "col": 34,
// JSON-NEXT:                     "tokLen": 1
// JSON-NEXT:                    },
// JSON-NEXT:                    "range": {
// JSON-NEXT:                     "begin": {
// JSON-NEXT:                      "offset": 794,
// JSON-NEXT:                      "col": 30,
// JSON-NEXT:                      "tokLen": 3
// JSON-NEXT:                     },
// JSON-NEXT:                     "end": {
// JSON-NEXT:                      "offset": 802,
// JSON-NEXT:                      "col": 38,
// JSON-NEXT:                      "tokLen": 1
// JSON-NEXT:                     }
// JSON-NEXT:                    },
// JSON-NEXT:                    "isReferenced": true,
// JSON-NEXT:                    "name": "Z",
// JSON-NEXT:                    "type": {
// JSON-NEXT:                     "qualType": "int"
// JSON-NEXT:                    },
// JSON-NEXT:                    "depth": 0,
// JSON-NEXT:                    "index": 2,
// JSON-NEXT:                    "defaultArg": {
// JSON-NEXT:                     "kind": "TemplateArgument",
// JSON-NEXT:                     "isExpr": true
// JSON-NEXT:                    },
// JSON-NEXT:                    "inner": [
// JSON-NEXT:                     {
// JSON-NEXT:                      "kind": "TemplateArgument",
// JSON-NEXT:                      "range": {
// JSON-NEXT:                       "begin": {
// JSON-NEXT:                        "offset": 802,
// JSON-NEXT:                        "col": 38,
// JSON-NEXT:                        "tokLen": 1
// JSON-NEXT:                       },
// JSON-NEXT:                       "end": {
// JSON-NEXT:                        "offset": 802,
// JSON-NEXT:                        "col": 38,
// JSON-NEXT:                        "tokLen": 1
// JSON-NEXT:                       }
// JSON-NEXT:                      },
// JSON-NEXT:                      "isExpr": true,
// JSON-NEXT:                      "inner": [
// JSON-NEXT:                       {
// JSON-NEXT:                        "id": "0x{{.*}}",
// JSON-NEXT:                        "kind": "IntegerLiteral",
// JSON-NEXT:                        "range": {
// JSON-NEXT:                         "begin": {
// JSON-NEXT:                          "offset": 802,
// JSON-NEXT:                          "col": 38,
// JSON-NEXT:                          "tokLen": 1
// JSON-NEXT:                         },
// JSON-NEXT:                         "end": {
// JSON-NEXT:                          "offset": 802,
// JSON-NEXT:                          "col": 38,
// JSON-NEXT:                          "tokLen": 1
// JSON-NEXT:                         }
// JSON-NEXT:                        },
// JSON-NEXT:                        "type": {
// JSON-NEXT:                         "qualType": "int"
// JSON-NEXT:                        },
// JSON-NEXT:                        "valueCategory": "prvalue",
// JSON-NEXT:                        "value": "5"
// JSON-NEXT:                       }
// JSON-NEXT:                      ]
// JSON-NEXT:                     }
// JSON-NEXT:                    ]
// JSON-NEXT:                   },
// JSON-NEXT:                   {
// JSON-NEXT:                    "id": "0x{{.*}}",
// JSON-NEXT:                    "kind": "IntegerLiteral",
// JSON-NEXT:                    "range": {
// JSON-NEXT:                     "begin": {
// JSON-NEXT:                      "offset": 873,
// JSON-NEXT:                      "line": 18,
// JSON-NEXT:                      "col": 29,
// JSON-NEXT:                      "tokLen": 1
// JSON-NEXT:                     },
// JSON-NEXT:                     "end": {
// JSON-NEXT:                      "offset": 873,
// JSON-NEXT:                      "col": 29,
// JSON-NEXT:                      "tokLen": 1
// JSON-NEXT:                     }
// JSON-NEXT:                    },
// JSON-NEXT:                    "type": {
// JSON-NEXT:                     "qualType": "int"
// JSON-NEXT:                    },
// JSON-NEXT:                    "valueCategory": "prvalue",
// JSON-NEXT:                    "value": "5"
// JSON-NEXT:                   }
// JSON-NEXT:                  ]
// JSON-NEXT:                 }
// JSON-NEXT:                ]
// JSON-NEXT:               }
// JSON-NEXT:              ]
// JSON-NEXT:             }
// JSON-NEXT:            ]
// JSON-NEXT:           }
// JSON-NEXT:          ]
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "CXXConstructorDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 812,
// JSON-NEXT:         "line": 15,
// JSON-NEXT:         "col": 8,
// JSON-NEXT:         "tokLen": 3
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 812,
// JSON-NEXT:          "col": 8,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 812,
// JSON-NEXT:          "col": 8,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "isImplicit": true,
// JSON-NEXT:        "name": "foo",
// JSON-NEXT:        "mangledName": "_ZN3fooILi5EiLi5EEC1ERKS0_",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "void (const foo<5, int> &)"
// JSON-NEXT:        },
// JSON-NEXT:        "inline": true,
// JSON-NEXT:        "constexpr": true,
// JSON-NEXT:        "explicitlyDefaulted": "default",
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "ParmVarDecl",
// JSON-NEXT:          "loc": {
// JSON-NEXT:           "offset": 812,
// JSON-NEXT:           "col": 8,
// JSON-NEXT:           "tokLen": 3
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 812,
// JSON-NEXT:            "col": 8,
// JSON-NEXT:            "tokLen": 3
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 812,
// JSON-NEXT:            "col": 8,
// JSON-NEXT:            "tokLen": 3
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "const foo<5, int> &"
// JSON-NEXT:          }
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "CXXConstructorDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 812,
// JSON-NEXT:         "col": 8,
// JSON-NEXT:         "tokLen": 3
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 812,
// JSON-NEXT:          "col": 8,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 812,
// JSON-NEXT:          "col": 8,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "isImplicit": true,
// JSON-NEXT:        "name": "foo",
// JSON-NEXT:        "mangledName": "_ZN3fooILi5EiLi5EEC1EOS0_",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "void (foo<5, int> &&)"
// JSON-NEXT:        },
// JSON-NEXT:        "inline": true,
// JSON-NEXT:        "constexpr": true,
// JSON-NEXT:        "explicitlyDefaulted": "default",
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "ParmVarDecl",
// JSON-NEXT:          "loc": {
// JSON-NEXT:           "offset": 812,
// JSON-NEXT:           "col": 8,
// JSON-NEXT:           "tokLen": 3
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 812,
// JSON-NEXT:            "col": 8,
// JSON-NEXT:            "tokLen": 3
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 812,
// JSON-NEXT:            "col": 8,
// JSON-NEXT:            "tokLen": 3
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "foo<5, int> &&"
// JSON-NEXT:          }
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "CXXDestructorDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 812,
// JSON-NEXT:         "col": 8,
// JSON-NEXT:         "tokLen": 3
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 812,
// JSON-NEXT:          "col": 8,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 812,
// JSON-NEXT:          "col": 8,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "isImplicit": true,
// JSON-NEXT:        "isReferenced": true,
// JSON-NEXT:        "name": "~foo",
// JSON-NEXT:        "mangledName": "_ZN3fooILi5EiLi5EED1Ev",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "void () noexcept"
// JSON-NEXT:        },
// JSON-NEXT:        "inline": true,
// JSON-NEXT:        "constexpr": true,
// JSON-NEXT:        "explicitlyDefaulted": "default"
// JSON-NEXT:       }
// JSON-NEXT:      ]
// JSON-NEXT:     },
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "ClassTemplateSpecializationDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": 812,
// JSON-NEXT:       "col": 8,
// JSON-NEXT:       "tokLen": 3
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 765,
// JSON-NEXT:        "line": 14,
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 879,
// JSON-NEXT:        "line": 19,
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "name": "foo",
// JSON-NEXT:      "tagUsed": "struct",
// JSON-NEXT:      "completeDefinition": true,
// JSON-NEXT:      "definitionData": {
// JSON-NEXT:       "canConstDefaultInit": true,
// JSON-NEXT:       "canPassInRegisters": true,
// JSON-NEXT:       "copyAssign": {
// JSON-NEXT:        "hasConstParam": true,
// JSON-NEXT:        "implicitHasConstParam": true,
// JSON-NEXT:        "needsImplicit": true,
// JSON-NEXT:        "simple": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       },
// JSON-NEXT:       "copyCtor": {
// JSON-NEXT:        "hasConstParam": true,
// JSON-NEXT:        "implicitHasConstParam": true,
// JSON-NEXT:        "simple": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       },
// JSON-NEXT:       "defaultCtor": {
// JSON-NEXT:        "defaultedIsConstexpr": true,
// JSON-NEXT:        "exists": true,
// JSON-NEXT:        "nonTrivial": true,
// JSON-NEXT:        "userProvided": true
// JSON-NEXT:       },
// JSON-NEXT:       "dtor": {
// JSON-NEXT:        "irrelevant": true,
// JSON-NEXT:        "simple": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       },
// JSON-NEXT:       "hasUserDeclaredConstructor": true,
// JSON-NEXT:       "isStandardLayout": true,
// JSON-NEXT:       "isTriviallyCopyable": true,
// JSON-NEXT:       "moveAssign": {
// JSON-NEXT:        "exists": true,
// JSON-NEXT:        "needsImplicit": true,
// JSON-NEXT:        "simple": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       },
// JSON-NEXT:       "moveCtor": {
// JSON-NEXT:        "exists": true,
// JSON-NEXT:        "simple": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "inner": [
// JSON-NEXT:       {
// JSON-NEXT:        "kind": "TemplateArgument",
// JSON-NEXT:        "value": 2
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "kind": "TemplateArgument",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "double"
// JSON-NEXT:        },
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "BuiltinType",
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "double"
// JSON-NEXT:          }
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "kind": "TemplateArgument",
// JSON-NEXT:        "value": 3
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "CXXRecordDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 812,
// JSON-NEXT:         "line": 15,
// JSON-NEXT:         "col": 8,
// JSON-NEXT:         "tokLen": 3
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 805,
// JSON-NEXT:          "col": 1,
// JSON-NEXT:          "tokLen": 6
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 812,
// JSON-NEXT:          "col": 8,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "isImplicit": true,
// JSON-NEXT:        "name": "foo",
// JSON-NEXT:        "tagUsed": "struct"
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "FieldDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 824,
// JSON-NEXT:         "line": 16,
// JSON-NEXT:         "col": 7,
// JSON-NEXT:         "tokLen": 8
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 820,
// JSON-NEXT:          "col": 3,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 824,
// JSON-NEXT:          "col": 7,
// JSON-NEXT:          "tokLen": 8
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "name": "constant",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "int"
// JSON-NEXT:        }
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "CXXConstructorDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 836,
// JSON-NEXT:         "line": 17,
// JSON-NEXT:         "col": 3,
// JSON-NEXT:         "tokLen": 3
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 836,
// JSON-NEXT:          "col": 3,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 843,
// JSON-NEXT:          "col": 10,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "isUsed": true,
// JSON-NEXT:        "name": "foo",
// JSON-NEXT:        "mangledName": "_ZN3fooILi2EdLi3EEC1Ev",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "void ()"
// JSON-NEXT:        },
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "CompoundStmt",
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 842,
// JSON-NEXT:            "col": 9,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 843,
// JSON-NEXT:            "col": 10,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          }
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "CXXMethodDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 849,
// JSON-NEXT:         "line": 18,
// JSON-NEXT:         "col": 5,
// JSON-NEXT:         "tokLen": 6
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 847,
// JSON-NEXT:          "col": 3,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 877,
// JSON-NEXT:          "col": 33,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "isUsed": true,
// JSON-NEXT:        "name": "getSum",
// JSON-NEXT:        "mangledName": "_ZN3fooILi2EdLi3EE6getSumEv",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "double ()"
// JSON-NEXT:        },
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "CompoundStmt",
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 858,
// JSON-NEXT:            "col": 14,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 877,
// JSON-NEXT:            "col": 33,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "inner": [
// JSON-NEXT:           {
// JSON-NEXT:            "id": "0x{{.*}}",
// JSON-NEXT:            "kind": "ReturnStmt",
// JSON-NEXT:            "range": {
// JSON-NEXT:             "begin": {
// JSON-NEXT:              "offset": 860,
// JSON-NEXT:              "col": 16,
// JSON-NEXT:              "tokLen": 6
// JSON-NEXT:             },
// JSON-NEXT:             "end": {
// JSON-NEXT:              "offset": 874,
// JSON-NEXT:              "col": 30,
// JSON-NEXT:              "tokLen": 1
// JSON-NEXT:             }
// JSON-NEXT:            },
// JSON-NEXT:            "inner": [
// JSON-NEXT:             {
// JSON-NEXT:              "id": "0x{{.*}}",
// JSON-NEXT:              "kind": "CXXFunctionalCastExpr",
// JSON-NEXT:              "range": {
// JSON-NEXT:               "begin": {
// JSON-NEXT:                "offset": 867,
// JSON-NEXT:                "col": 23,
// JSON-NEXT:                "tokLen": 1
// JSON-NEXT:               },
// JSON-NEXT:               "end": {
// JSON-NEXT:                "offset": 874,
// JSON-NEXT:                "col": 30,
// JSON-NEXT:                "tokLen": 1
// JSON-NEXT:               }
// JSON-NEXT:              },
// JSON-NEXT:              "type": {
// JSON-NEXT:               "qualType": "double"
// JSON-NEXT:              },
// JSON-NEXT:              "valueCategory": "prvalue",
// JSON-NEXT:              "castKind": "NoOp",
// JSON-NEXT:              "inner": [
// JSON-NEXT:               {
// JSON-NEXT:                "id": "0x{{.*}}",
// JSON-NEXT:                "kind": "ImplicitCastExpr",
// JSON-NEXT:                "range": {
// JSON-NEXT:                 "begin": {
// JSON-NEXT:                  "offset": 869,
// JSON-NEXT:                  "col": 25,
// JSON-NEXT:                  "tokLen": 1
// JSON-NEXT:                 },
// JSON-NEXT:                 "end": {
// JSON-NEXT:                  "offset": 873,
// JSON-NEXT:                  "col": 29,
// JSON-NEXT:                  "tokLen": 1
// JSON-NEXT:                 }
// JSON-NEXT:                },
// JSON-NEXT:                "type": {
// JSON-NEXT:                 "qualType": "double"
// JSON-NEXT:                },
// JSON-NEXT:                "valueCategory": "prvalue",
// JSON-NEXT:                "castKind": "IntegralToFloating",
// JSON-NEXT:                "isPartOfExplicitCast": true,
// JSON-NEXT:                "inner": [
// JSON-NEXT:                 {
// JSON-NEXT:                  "id": "0x{{.*}}",
// JSON-NEXT:                  "kind": "BinaryOperator",
// JSON-NEXT:                  "range": {
// JSON-NEXT:                   "begin": {
// JSON-NEXT:                    "offset": 869,
// JSON-NEXT:                    "col": 25,
// JSON-NEXT:                    "tokLen": 1
// JSON-NEXT:                   },
// JSON-NEXT:                   "end": {
// JSON-NEXT:                    "offset": 873,
// JSON-NEXT:                    "col": 29,
// JSON-NEXT:                    "tokLen": 1
// JSON-NEXT:                   }
// JSON-NEXT:                  },
// JSON-NEXT:                  "type": {
// JSON-NEXT:                   "qualType": "int"
// JSON-NEXT:                  },
// JSON-NEXT:                  "valueCategory": "prvalue",
// JSON-NEXT:                  "opcode": "+",
// JSON-NEXT:                  "inner": [
// JSON-NEXT:                   {
// JSON-NEXT:                    "id": "0x{{.*}}",
// JSON-NEXT:                    "kind": "SubstNonTypeTemplateParmExpr",
// JSON-NEXT:                    "range": {
// JSON-NEXT:                     "begin": {
// JSON-NEXT:                      "offset": 869,
// JSON-NEXT:                      "col": 25,
// JSON-NEXT:                      "tokLen": 1
// JSON-NEXT:                     },
// JSON-NEXT:                     "end": {
// JSON-NEXT:                      "offset": 869,
// JSON-NEXT:                      "col": 25,
// JSON-NEXT:                      "tokLen": 1
// JSON-NEXT:                     }
// JSON-NEXT:                    },
// JSON-NEXT:                    "type": {
// JSON-NEXT:                     "qualType": "int"
// JSON-NEXT:                    },
// JSON-NEXT:                    "valueCategory": "prvalue",
// JSON-NEXT:                    "inner": [
// JSON-NEXT:                     {
// JSON-NEXT:                      "id": "0x{{.*}}",
// JSON-NEXT:                      "kind": "NonTypeTemplateParmDecl",
// JSON-NEXT:                      "loc": {
// JSON-NEXT:                       "offset": 779,
// JSON-NEXT:                       "line": 14,
// JSON-NEXT:                       "col": 15,
// JSON-NEXT:                       "tokLen": 1
// JSON-NEXT:                      },
// JSON-NEXT:                      "range": {
// JSON-NEXT:                       "begin": {
// JSON-NEXT:                        "offset": 775,
// JSON-NEXT:                        "col": 11,
// JSON-NEXT:                        "tokLen": 3
// JSON-NEXT:                       },
// JSON-NEXT:                       "end": {
// JSON-NEXT:                        "offset": 779,
// JSON-NEXT:                        "col": 15,
// JSON-NEXT:                        "tokLen": 1
// JSON-NEXT:                       }
// JSON-NEXT:                      },
// JSON-NEXT:                      "isReferenced": true,
// JSON-NEXT:                      "name": "X",
// JSON-NEXT:                      "type": {
// JSON-NEXT:                       "qualType": "int"
// JSON-NEXT:                      },
// JSON-NEXT:                      "depth": 0,
// JSON-NEXT:                      "index": 0
// JSON-NEXT:                     },
// JSON-NEXT:                     {
// JSON-NEXT:                      "id": "0x{{.*}}",
// JSON-NEXT:                      "kind": "IntegerLiteral",
// JSON-NEXT:                      "range": {
// JSON-NEXT:                       "begin": {
// JSON-NEXT:                        "offset": 869,
// JSON-NEXT:                        "line": 18,
// JSON-NEXT:                        "col": 25,
// JSON-NEXT:                        "tokLen": 1
// JSON-NEXT:                       },
// JSON-NEXT:                       "end": {
// JSON-NEXT:                        "offset": 869,
// JSON-NEXT:                        "col": 25,
// JSON-NEXT:                        "tokLen": 1
// JSON-NEXT:                       }
// JSON-NEXT:                      },
// JSON-NEXT:                      "type": {
// JSON-NEXT:                       "qualType": "int"
// JSON-NEXT:                      },
// JSON-NEXT:                      "valueCategory": "prvalue",
// JSON-NEXT:                      "value": "2"
// JSON-NEXT:                     }
// JSON-NEXT:                    ]
// JSON-NEXT:                   },
// JSON-NEXT:                   {
// JSON-NEXT:                    "id": "0x{{.*}}",
// JSON-NEXT:                    "kind": "SubstNonTypeTemplateParmExpr",
// JSON-NEXT:                    "range": {
// JSON-NEXT:                     "begin": {
// JSON-NEXT:                      "offset": 873,
// JSON-NEXT:                      "col": 29,
// JSON-NEXT:                      "tokLen": 1
// JSON-NEXT:                     },
// JSON-NEXT:                     "end": {
// JSON-NEXT:                      "offset": 873,
// JSON-NEXT:                      "col": 29,
// JSON-NEXT:                      "tokLen": 1
// JSON-NEXT:                     }
// JSON-NEXT:                    },
// JSON-NEXT:                    "type": {
// JSON-NEXT:                     "qualType": "int"
// JSON-NEXT:                    },
// JSON-NEXT:                    "valueCategory": "prvalue",
// JSON-NEXT:                    "inner": [
// JSON-NEXT:                     {
// JSON-NEXT:                      "id": "0x{{.*}}",
// JSON-NEXT:                      "kind": "NonTypeTemplateParmDecl",
// JSON-NEXT:                      "loc": {
// JSON-NEXT:                       "offset": 798,
// JSON-NEXT:                       "line": 14,
// JSON-NEXT:                       "col": 34,
// JSON-NEXT:                       "tokLen": 1
// JSON-NEXT:                      },
// JSON-NEXT:                      "range": {
// JSON-NEXT:                       "begin": {
// JSON-NEXT:                        "offset": 794,
// JSON-NEXT:                        "col": 30,
// JSON-NEXT:                        "tokLen": 3
// JSON-NEXT:                       },
// JSON-NEXT:                       "end": {
// JSON-NEXT:                        "offset": 802,
// JSON-NEXT:                        "col": 38,
// JSON-NEXT:                        "tokLen": 1
// JSON-NEXT:                       }
// JSON-NEXT:                      },
// JSON-NEXT:                      "isReferenced": true,
// JSON-NEXT:                      "name": "Z",
// JSON-NEXT:                      "type": {
// JSON-NEXT:                       "qualType": "int"
// JSON-NEXT:                      },
// JSON-NEXT:                      "depth": 0,
// JSON-NEXT:                      "index": 2,
// JSON-NEXT:                      "defaultArg": {
// JSON-NEXT:                       "kind": "TemplateArgument",
// JSON-NEXT:                       "isExpr": true
// JSON-NEXT:                      },
// JSON-NEXT:                      "inner": [
// JSON-NEXT:                       {
// JSON-NEXT:                        "kind": "TemplateArgument",
// JSON-NEXT:                        "range": {
// JSON-NEXT:                         "begin": {
// JSON-NEXT:                          "offset": 802,
// JSON-NEXT:                          "col": 38,
// JSON-NEXT:                          "tokLen": 1
// JSON-NEXT:                         },
// JSON-NEXT:                         "end": {
// JSON-NEXT:                          "offset": 802,
// JSON-NEXT:                          "col": 38,
// JSON-NEXT:                          "tokLen": 1
// JSON-NEXT:                         }
// JSON-NEXT:                        },
// JSON-NEXT:                        "isExpr": true,
// JSON-NEXT:                        "inner": [
// JSON-NEXT:                         {
// JSON-NEXT:                          "id": "0x{{.*}}",
// JSON-NEXT:                          "kind": "IntegerLiteral",
// JSON-NEXT:                          "range": {
// JSON-NEXT:                           "begin": {
// JSON-NEXT:                            "offset": 802,
// JSON-NEXT:                            "col": 38,
// JSON-NEXT:                            "tokLen": 1
// JSON-NEXT:                           },
// JSON-NEXT:                           "end": {
// JSON-NEXT:                            "offset": 802,
// JSON-NEXT:                            "col": 38,
// JSON-NEXT:                            "tokLen": 1
// JSON-NEXT:                           }
// JSON-NEXT:                          },
// JSON-NEXT:                          "type": {
// JSON-NEXT:                           "qualType": "int"
// JSON-NEXT:                          },
// JSON-NEXT:                          "valueCategory": "prvalue",
// JSON-NEXT:                          "value": "5"
// JSON-NEXT:                         }
// JSON-NEXT:                        ]
// JSON-NEXT:                       }
// JSON-NEXT:                      ]
// JSON-NEXT:                     },
// JSON-NEXT:                     {
// JSON-NEXT:                      "id": "0x{{.*}}",
// JSON-NEXT:                      "kind": "IntegerLiteral",
// JSON-NEXT:                      "range": {
// JSON-NEXT:                       "begin": {
// JSON-NEXT:                        "offset": 873,
// JSON-NEXT:                        "line": 18,
// JSON-NEXT:                        "col": 29,
// JSON-NEXT:                        "tokLen": 1
// JSON-NEXT:                       },
// JSON-NEXT:                       "end": {
// JSON-NEXT:                        "offset": 873,
// JSON-NEXT:                        "col": 29,
// JSON-NEXT:                        "tokLen": 1
// JSON-NEXT:                       }
// JSON-NEXT:                      },
// JSON-NEXT:                      "type": {
// JSON-NEXT:                       "qualType": "int"
// JSON-NEXT:                      },
// JSON-NEXT:                      "valueCategory": "prvalue",
// JSON-NEXT:                      "value": "3"
// JSON-NEXT:                     }
// JSON-NEXT:                    ]
// JSON-NEXT:                   }
// JSON-NEXT:                  ]
// JSON-NEXT:                 }
// JSON-NEXT:                ]
// JSON-NEXT:               }
// JSON-NEXT:              ]
// JSON-NEXT:             }
// JSON-NEXT:            ]
// JSON-NEXT:           }
// JSON-NEXT:          ]
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "CXXConstructorDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 812,
// JSON-NEXT:         "line": 15,
// JSON-NEXT:         "col": 8,
// JSON-NEXT:         "tokLen": 3
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 812,
// JSON-NEXT:          "col": 8,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 812,
// JSON-NEXT:          "col": 8,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "isImplicit": true,
// JSON-NEXT:        "name": "foo",
// JSON-NEXT:        "mangledName": "_ZN3fooILi2EdLi3EEC1ERKS0_",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "void (const foo<2, double, 3> &)"
// JSON-NEXT:        },
// JSON-NEXT:        "inline": true,
// JSON-NEXT:        "constexpr": true,
// JSON-NEXT:        "explicitlyDefaulted": "default",
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "ParmVarDecl",
// JSON-NEXT:          "loc": {
// JSON-NEXT:           "offset": 812,
// JSON-NEXT:           "col": 8,
// JSON-NEXT:           "tokLen": 3
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 812,
// JSON-NEXT:            "col": 8,
// JSON-NEXT:            "tokLen": 3
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 812,
// JSON-NEXT:            "col": 8,
// JSON-NEXT:            "tokLen": 3
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "const foo<2, double, 3> &"
// JSON-NEXT:          }
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "CXXConstructorDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 812,
// JSON-NEXT:         "col": 8,
// JSON-NEXT:         "tokLen": 3
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 812,
// JSON-NEXT:          "col": 8,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 812,
// JSON-NEXT:          "col": 8,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "isImplicit": true,
// JSON-NEXT:        "name": "foo",
// JSON-NEXT:        "mangledName": "_ZN3fooILi2EdLi3EEC1EOS0_",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "void (foo<2, double, 3> &&)"
// JSON-NEXT:        },
// JSON-NEXT:        "inline": true,
// JSON-NEXT:        "constexpr": true,
// JSON-NEXT:        "explicitlyDefaulted": "default",
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "ParmVarDecl",
// JSON-NEXT:          "loc": {
// JSON-NEXT:           "offset": 812,
// JSON-NEXT:           "col": 8,
// JSON-NEXT:           "tokLen": 3
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 812,
// JSON-NEXT:            "col": 8,
// JSON-NEXT:            "tokLen": 3
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 812,
// JSON-NEXT:            "col": 8,
// JSON-NEXT:            "tokLen": 3
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "foo<2, double, 3> &&"
// JSON-NEXT:          }
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "CXXDestructorDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 812,
// JSON-NEXT:         "col": 8,
// JSON-NEXT:         "tokLen": 3
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 812,
// JSON-NEXT:          "col": 8,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 812,
// JSON-NEXT:          "col": 8,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "isImplicit": true,
// JSON-NEXT:        "isReferenced": true,
// JSON-NEXT:        "name": "~foo",
// JSON-NEXT:        "mangledName": "_ZN3fooILi2EdLi3EED1Ev",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "void () noexcept"
// JSON-NEXT:        },
// JSON-NEXT:        "inline": true,
// JSON-NEXT:        "constexpr": true,
// JSON-NEXT:        "explicitlyDefaulted": "default"
// JSON-NEXT:       }
// JSON-NEXT:      ]
// JSON-NEXT:     }
// JSON-NEXT:    ]
// JSON-NEXT:   },
// JSON-NEXT:   {
// JSON-NEXT:    "id": "0x{{.*}}",
// JSON-NEXT:    "kind": "FunctionTemplateDecl",
// JSON-NEXT:    "loc": {
// JSON-NEXT:     "offset": 914,
// JSON-NEXT:     "line": 22,
// JSON-NEXT:     "col": 3,
// JSON-NEXT:     "tokLen": 3
// JSON-NEXT:    },
// JSON-NEXT:    "range": {
// JSON-NEXT:     "begin": {
// JSON-NEXT:      "offset": 883,
// JSON-NEXT:      "line": 21,
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 8
// JSON-NEXT:     },
// JSON-NEXT:     "end": {
// JSON-NEXT:      "offset": 937,
// JSON-NEXT:      "line": 24,
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 1
// JSON-NEXT:     }
// JSON-NEXT:    },
// JSON-NEXT:    "name": "bar",
// JSON-NEXT:    "inner": [
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "NonTypeTemplateParmDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": 897,
// JSON-NEXT:       "line": 21,
// JSON-NEXT:       "col": 15,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 893,
// JSON-NEXT:        "col": 11,
// JSON-NEXT:        "tokLen": 3
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 897,
// JSON-NEXT:        "col": 15,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "isReferenced": true,
// JSON-NEXT:      "name": "A",
// JSON-NEXT:      "type": {
// JSON-NEXT:       "qualType": "int"
// JSON-NEXT:      },
// JSON-NEXT:      "depth": 0,
// JSON-NEXT:      "index": 0
// JSON-NEXT:     },
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "TemplateTypeParmDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": 909,
// JSON-NEXT:       "col": 27,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 900,
// JSON-NEXT:        "col": 18,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 909,
// JSON-NEXT:        "col": 27,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "isReferenced": true,
// JSON-NEXT:      "name": "B",
// JSON-NEXT:      "tagUsed": "typename",
// JSON-NEXT:      "depth": 0,
// JSON-NEXT:      "index": 1
// JSON-NEXT:     },
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "FunctionDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": 914,
// JSON-NEXT:       "line": 22,
// JSON-NEXT:       "col": 3,
// JSON-NEXT:       "tokLen": 3
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 912,
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 937,
// JSON-NEXT:        "line": 24,
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "name": "bar",
// JSON-NEXT:      "type": {
// JSON-NEXT:       "qualType": "B ()"
// JSON-NEXT:      },
// JSON-NEXT:      "inner": [
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "CompoundStmt",
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 920,
// JSON-NEXT:          "line": 22,
// JSON-NEXT:          "col": 9,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 937,
// JSON-NEXT:          "line": 24,
// JSON-NEXT:          "col": 1,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "ReturnStmt",
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 924,
// JSON-NEXT:            "line": 23,
// JSON-NEXT:            "col": 3,
// JSON-NEXT:            "tokLen": 6
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 934,
// JSON-NEXT:            "col": 13,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "inner": [
// JSON-NEXT:           {
// JSON-NEXT:            "id": "0x{{.*}}",
// JSON-NEXT:            "kind": "CXXUnresolvedConstructExpr",
// JSON-NEXT:            "range": {
// JSON-NEXT:             "begin": {
// JSON-NEXT:              "offset": 931,
// JSON-NEXT:              "col": 10,
// JSON-NEXT:              "tokLen": 1
// JSON-NEXT:             },
// JSON-NEXT:             "end": {
// JSON-NEXT:              "offset": 934,
// JSON-NEXT:              "col": 13,
// JSON-NEXT:              "tokLen": 1
// JSON-NEXT:             }
// JSON-NEXT:            },
// JSON-NEXT:            "type": {
// JSON-NEXT:             "qualType": "B"
// JSON-NEXT:            },
// JSON-NEXT:            "valueCategory": "prvalue",
// JSON-NEXT:            "inner": [
// JSON-NEXT:             {
// JSON-NEXT:              "id": "0x{{.*}}",
// JSON-NEXT:              "kind": "DeclRefExpr",
// JSON-NEXT:              "range": {
// JSON-NEXT:               "begin": {
// JSON-NEXT:                "offset": 933,
// JSON-NEXT:                "col": 12,
// JSON-NEXT:                "tokLen": 1
// JSON-NEXT:               },
// JSON-NEXT:               "end": {
// JSON-NEXT:                "offset": 933,
// JSON-NEXT:                "col": 12,
// JSON-NEXT:                "tokLen": 1
// JSON-NEXT:               }
// JSON-NEXT:              },
// JSON-NEXT:              "type": {
// JSON-NEXT:               "qualType": "int"
// JSON-NEXT:              },
// JSON-NEXT:              "valueCategory": "prvalue",
// JSON-NEXT:              "referencedDecl": {
// JSON-NEXT:               "id": "0x{{.*}}",
// JSON-NEXT:               "kind": "NonTypeTemplateParmDecl",
// JSON-NEXT:               "name": "A",
// JSON-NEXT:               "type": {
// JSON-NEXT:                "qualType": "int"
// JSON-NEXT:               }
// JSON-NEXT:              }
// JSON-NEXT:             }
// JSON-NEXT:            ]
// JSON-NEXT:           }
// JSON-NEXT:          ]
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       }
// JSON-NEXT:      ]
// JSON-NEXT:     },
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "FunctionDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": 914,
// JSON-NEXT:       "line": 22,
// JSON-NEXT:       "col": 3,
// JSON-NEXT:       "tokLen": 3
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 912,
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 937,
// JSON-NEXT:        "line": 24,
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "isUsed": true,
// JSON-NEXT:      "name": "bar",
// JSON-NEXT:      "mangledName": "_Z3barILi5EiET0_v",
// JSON-NEXT:      "type": {
// JSON-NEXT:       "qualType": "int ()"
// JSON-NEXT:      },
// JSON-NEXT:      "inner": [
// JSON-NEXT:       {
// JSON-NEXT:        "kind": "TemplateArgument",
// JSON-NEXT:        "value": 5
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "kind": "TemplateArgument",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "int"
// JSON-NEXT:        },
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "BuiltinType",
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "int"
// JSON-NEXT:          }
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "CompoundStmt",
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 920,
// JSON-NEXT:          "line": 22,
// JSON-NEXT:          "col": 9,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 937,
// JSON-NEXT:          "line": 24,
// JSON-NEXT:          "col": 1,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "ReturnStmt",
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 924,
// JSON-NEXT:            "line": 23,
// JSON-NEXT:            "col": 3,
// JSON-NEXT:            "tokLen": 6
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 934,
// JSON-NEXT:            "col": 13,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "inner": [
// JSON-NEXT:           {
// JSON-NEXT:            "id": "0x{{.*}}",
// JSON-NEXT:            "kind": "CXXFunctionalCastExpr",
// JSON-NEXT:            "range": {
// JSON-NEXT:             "begin": {
// JSON-NEXT:              "offset": 931,
// JSON-NEXT:              "col": 10,
// JSON-NEXT:              "tokLen": 1
// JSON-NEXT:             },
// JSON-NEXT:             "end": {
// JSON-NEXT:              "offset": 934,
// JSON-NEXT:              "col": 13,
// JSON-NEXT:              "tokLen": 1
// JSON-NEXT:             }
// JSON-NEXT:            },
// JSON-NEXT:            "type": {
// JSON-NEXT:             "qualType": "int"
// JSON-NEXT:            },
// JSON-NEXT:            "valueCategory": "prvalue",
// JSON-NEXT:            "castKind": "NoOp",
// JSON-NEXT:            "inner": [
// JSON-NEXT:             {
// JSON-NEXT:              "id": "0x{{.*}}",
// JSON-NEXT:              "kind": "SubstNonTypeTemplateParmExpr",
// JSON-NEXT:              "range": {
// JSON-NEXT:               "begin": {
// JSON-NEXT:                "offset": 933,
// JSON-NEXT:                "col": 12,
// JSON-NEXT:                "tokLen": 1
// JSON-NEXT:               },
// JSON-NEXT:               "end": {
// JSON-NEXT:                "offset": 933,
// JSON-NEXT:                "col": 12,
// JSON-NEXT:                "tokLen": 1
// JSON-NEXT:               }
// JSON-NEXT:              },
// JSON-NEXT:              "type": {
// JSON-NEXT:               "qualType": "int"
// JSON-NEXT:              },
// JSON-NEXT:              "valueCategory": "prvalue",
// JSON-NEXT:              "inner": [
// JSON-NEXT:               {
// JSON-NEXT:                "id": "0x{{.*}}",
// JSON-NEXT:                "kind": "NonTypeTemplateParmDecl",
// JSON-NEXT:                "loc": {
// JSON-NEXT:                 "offset": 897,
// JSON-NEXT:                 "line": 21,
// JSON-NEXT:                 "col": 15,
// JSON-NEXT:                 "tokLen": 1
// JSON-NEXT:                },
// JSON-NEXT:                "range": {
// JSON-NEXT:                 "begin": {
// JSON-NEXT:                  "offset": 893,
// JSON-NEXT:                  "col": 11,
// JSON-NEXT:                  "tokLen": 3
// JSON-NEXT:                 },
// JSON-NEXT:                 "end": {
// JSON-NEXT:                  "offset": 897,
// JSON-NEXT:                  "col": 15,
// JSON-NEXT:                  "tokLen": 1
// JSON-NEXT:                 }
// JSON-NEXT:                },
// JSON-NEXT:                "isReferenced": true,
// JSON-NEXT:                "name": "A",
// JSON-NEXT:                "type": {
// JSON-NEXT:                 "qualType": "int"
// JSON-NEXT:                },
// JSON-NEXT:                "depth": 0,
// JSON-NEXT:                "index": 0
// JSON-NEXT:               },
// JSON-NEXT:               {
// JSON-NEXT:                "id": "0x{{.*}}",
// JSON-NEXT:                "kind": "IntegerLiteral",
// JSON-NEXT:                "range": {
// JSON-NEXT:                 "begin": {
// JSON-NEXT:                  "offset": 933,
// JSON-NEXT:                  "line": 23,
// JSON-NEXT:                  "col": 12,
// JSON-NEXT:                  "tokLen": 1
// JSON-NEXT:                 },
// JSON-NEXT:                 "end": {
// JSON-NEXT:                  "offset": 933,
// JSON-NEXT:                  "col": 12,
// JSON-NEXT:                  "tokLen": 1
// JSON-NEXT:                 }
// JSON-NEXT:                },
// JSON-NEXT:                "type": {
// JSON-NEXT:                 "qualType": "int"
// JSON-NEXT:                },
// JSON-NEXT:                "valueCategory": "prvalue",
// JSON-NEXT:                "value": "5"
// JSON-NEXT:               }
// JSON-NEXT:              ]
// JSON-NEXT:             }
// JSON-NEXT:            ]
// JSON-NEXT:           }
// JSON-NEXT:          ]
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       }
// JSON-NEXT:      ]
// JSON-NEXT:     }
// JSON-NEXT:    ]
// JSON-NEXT:   },
// JSON-NEXT:   {
// JSON-NEXT:    "id": "0x{{.*}}",
// JSON-NEXT:    "kind": "FunctionDecl",
// JSON-NEXT:    "loc": {
// JSON-NEXT:     "offset": 945,
// JSON-NEXT:     "line": 26,
// JSON-NEXT:     "col": 6,
// JSON-NEXT:     "tokLen": 3
// JSON-NEXT:    },
// JSON-NEXT:    "range": {
// JSON-NEXT:     "begin": {
// JSON-NEXT:      "offset": 940,
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 4
// JSON-NEXT:     },
// JSON-NEXT:     "end": {
// JSON-NEXT:      "offset": 1055,
// JSON-NEXT:      "line": 30,
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 1
// JSON-NEXT:     }
// JSON-NEXT:    },
// JSON-NEXT:    "name": "baz",
// JSON-NEXT:    "mangledName": "_Z3bazv",
// JSON-NEXT:    "type": {
// JSON-NEXT:     "qualType": "void ()"
// JSON-NEXT:    },
// JSON-NEXT:    "inner": [
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "CompoundStmt",
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 951,
// JSON-NEXT:        "line": 26,
// JSON-NEXT:        "col": 12,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 1055,
// JSON-NEXT:        "line": 30,
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "inner": [
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "DeclStmt",
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 955,
// JSON-NEXT:          "line": 27,
// JSON-NEXT:          "col": 3,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 976,
// JSON-NEXT:          "col": 24,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "VarDecl",
// JSON-NEXT:          "loc": {
// JSON-NEXT:           "offset": 959,
// JSON-NEXT:           "col": 7,
// JSON-NEXT:           "tokLen": 1
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 955,
// JSON-NEXT:            "col": 3,
// JSON-NEXT:            "tokLen": 3
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 975,
// JSON-NEXT:            "col": 23,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "name": "x",
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "int"
// JSON-NEXT:          },
// JSON-NEXT:          "init": "c",
// JSON-NEXT:          "inner": [
// JSON-NEXT:           {
// JSON-NEXT:            "id": "0x{{.*}}",
// JSON-NEXT:            "kind": "CallExpr",
// JSON-NEXT:            "range": {
// JSON-NEXT:             "begin": {
// JSON-NEXT:              "offset": 963,
// JSON-NEXT:              "col": 11,
// JSON-NEXT:              "tokLen": 3
// JSON-NEXT:             },
// JSON-NEXT:             "end": {
// JSON-NEXT:              "offset": 975,
// JSON-NEXT:              "col": 23,
// JSON-NEXT:              "tokLen": 1
// JSON-NEXT:             }
// JSON-NEXT:            },
// JSON-NEXT:            "type": {
// JSON-NEXT:             "qualType": "int"
// JSON-NEXT:            },
// JSON-NEXT:            "valueCategory": "prvalue",
// JSON-NEXT:            "inner": [
// JSON-NEXT:             {
// JSON-NEXT:              "id": "0x{{.*}}",
// JSON-NEXT:              "kind": "ImplicitCastExpr",
// JSON-NEXT:              "range": {
// JSON-NEXT:               "begin": {
// JSON-NEXT:                "offset": 963,
// JSON-NEXT:                "col": 11,
// JSON-NEXT:                "tokLen": 3
// JSON-NEXT:               },
// JSON-NEXT:               "end": {
// JSON-NEXT:                "offset": 973,
// JSON-NEXT:                "col": 21,
// JSON-NEXT:                "tokLen": 1
// JSON-NEXT:               }
// JSON-NEXT:              },
// JSON-NEXT:              "type": {
// JSON-NEXT:               "qualType": "int (*)()"
// JSON-NEXT:              },
// JSON-NEXT:              "valueCategory": "prvalue",
// JSON-NEXT:              "castKind": "FunctionToPointerDecay",
// JSON-NEXT:              "inner": [
// JSON-NEXT:               {
// JSON-NEXT:                "id": "0x{{.*}}",
// JSON-NEXT:                "kind": "DeclRefExpr",
// JSON-NEXT:                "range": {
// JSON-NEXT:                 "begin": {
// JSON-NEXT:                  "offset": 963,
// JSON-NEXT:                  "col": 11,
// JSON-NEXT:                  "tokLen": 3
// JSON-NEXT:                 },
// JSON-NEXT:                 "end": {
// JSON-NEXT:                  "offset": 973,
// JSON-NEXT:                  "col": 21,
// JSON-NEXT:                  "tokLen": 1
// JSON-NEXT:                 }
// JSON-NEXT:                },
// JSON-NEXT:                "type": {
// JSON-NEXT:                 "qualType": "int ()"
// JSON-NEXT:                },
// JSON-NEXT:                "valueCategory": "lvalue",
// JSON-NEXT:                "referencedDecl": {
// JSON-NEXT:                 "id": "0x{{.*}}",
// JSON-NEXT:                 "kind": "FunctionDecl",
// JSON-NEXT:                 "name": "bar",
// JSON-NEXT:                 "type": {
// JSON-NEXT:                  "qualType": "int ()"
// JSON-NEXT:                 }
// JSON-NEXT:                },
// JSON-NEXT:                "foundReferencedDecl": {
// JSON-NEXT:                 "id": "0x{{.*}}",
// JSON-NEXT:                 "kind": "FunctionTemplateDecl",
// JSON-NEXT:                 "name": "bar"
// JSON-NEXT:                }
// JSON-NEXT:               }
// JSON-NEXT:              ]
// JSON-NEXT:             }
// JSON-NEXT:            ]
// JSON-NEXT:           }
// JSON-NEXT:          ]
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "DeclStmt",
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 980,
// JSON-NEXT:          "line": 28,
// JSON-NEXT:          "col": 3,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 1010,
// JSON-NEXT:          "col": 33,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "VarDecl",
// JSON-NEXT:          "loc": {
// JSON-NEXT:           "offset": 984,
// JSON-NEXT:           "col": 7,
// JSON-NEXT:           "tokLen": 1
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 980,
// JSON-NEXT:            "col": 3,
// JSON-NEXT:            "tokLen": 3
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 1009,
// JSON-NEXT:            "col": 32,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "name": "y",
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "int"
// JSON-NEXT:          },
// JSON-NEXT:          "init": "c",
// JSON-NEXT:          "inner": [
// JSON-NEXT:           {
// JSON-NEXT:            "id": "0x{{.*}}",
// JSON-NEXT:            "kind": "ExprWithCleanups",
// JSON-NEXT:            "range": {
// JSON-NEXT:             "begin": {
// JSON-NEXT:              "offset": 988,
// JSON-NEXT:              "col": 11,
// JSON-NEXT:              "tokLen": 3
// JSON-NEXT:             },
// JSON-NEXT:             "end": {
// JSON-NEXT:              "offset": 1009,
// JSON-NEXT:              "col": 32,
// JSON-NEXT:              "tokLen": 1
// JSON-NEXT:             }
// JSON-NEXT:            },
// JSON-NEXT:            "type": {
// JSON-NEXT:             "qualType": "int"
// JSON-NEXT:            },
// JSON-NEXT:            "valueCategory": "prvalue",
// JSON-NEXT:            "inner": [
// JSON-NEXT:             {
// JSON-NEXT:              "id": "0x{{.*}}",
// JSON-NEXT:              "kind": "CXXMemberCallExpr",
// JSON-NEXT:              "range": {
// JSON-NEXT:               "begin": {
// JSON-NEXT:                "offset": 988,
// JSON-NEXT:                "col": 11,
// JSON-NEXT:                "tokLen": 3
// JSON-NEXT:               },
// JSON-NEXT:               "end": {
// JSON-NEXT:                "offset": 1009,
// JSON-NEXT:                "col": 32,
// JSON-NEXT:                "tokLen": 1
// JSON-NEXT:               }
// JSON-NEXT:              },
// JSON-NEXT:              "type": {
// JSON-NEXT:               "qualType": "int"
// JSON-NEXT:              },
// JSON-NEXT:              "valueCategory": "prvalue",
// JSON-NEXT:              "inner": [
// JSON-NEXT:               {
// JSON-NEXT:                "id": "0x{{.*}}",
// JSON-NEXT:                "kind": "MemberExpr",
// JSON-NEXT:                "range": {
// JSON-NEXT:                 "begin": {
// JSON-NEXT:                  "offset": 988,
// JSON-NEXT:                  "col": 11,
// JSON-NEXT:                  "tokLen": 3
// JSON-NEXT:                 },
// JSON-NEXT:                 "end": {
// JSON-NEXT:                  "offset": 1002,
// JSON-NEXT:                  "col": 25,
// JSON-NEXT:                  "tokLen": 6
// JSON-NEXT:                 }
// JSON-NEXT:                },
// JSON-NEXT:                "type": {
// JSON-NEXT:                 "qualType": "<bound member function type>"
// JSON-NEXT:                },
// JSON-NEXT:                "valueCategory": "prvalue",
// JSON-NEXT:                "name": "getSum",
// JSON-NEXT:                "isArrow": false,
// JSON-NEXT:                "referencedMemberDecl": "0x{{.*}}",
// JSON-NEXT:                "inner": [
// JSON-NEXT:                 {
// JSON-NEXT:                  "id": "0x{{.*}}",
// JSON-NEXT:                  "kind": "MaterializeTemporaryExpr",
// JSON-NEXT:                  "range": {
// JSON-NEXT:                   "begin": {
// JSON-NEXT:                    "offset": 988,
// JSON-NEXT:                    "col": 11,
// JSON-NEXT:                    "tokLen": 3
// JSON-NEXT:                   },
// JSON-NEXT:                   "end": {
// JSON-NEXT:                    "offset": 1000,
// JSON-NEXT:                    "col": 23,
// JSON-NEXT:                    "tokLen": 1
// JSON-NEXT:                   }
// JSON-NEXT:                  },
// JSON-NEXT:                  "type": {
// JSON-NEXT:                   "qualType": "foo<5, int>"
// JSON-NEXT:                  },
// JSON-NEXT:                  "valueCategory": "xvalue",
// JSON-NEXT:                  "storageDuration": "full expression",
// JSON-NEXT:                  "inner": [
// JSON-NEXT:                   {
// JSON-NEXT:                    "id": "0x{{.*}}",
// JSON-NEXT:                    "kind": "CXXTemporaryObjectExpr",
// JSON-NEXT:                    "range": {
// JSON-NEXT:                     "begin": {
// JSON-NEXT:                      "offset": 988,
// JSON-NEXT:                      "col": 11,
// JSON-NEXT:                      "tokLen": 3
// JSON-NEXT:                     },
// JSON-NEXT:                     "end": {
// JSON-NEXT:                      "offset": 1000,
// JSON-NEXT:                      "col": 23,
// JSON-NEXT:                      "tokLen": 1
// JSON-NEXT:                     }
// JSON-NEXT:                    },
// JSON-NEXT:                    "type": {
// JSON-NEXT:                     "qualType": "foo<5, int>"
// JSON-NEXT:                    },
// JSON-NEXT:                    "valueCategory": "prvalue",
// JSON-NEXT:                    "ctorType": {
// JSON-NEXT:                     "qualType": "void ()"
// JSON-NEXT:                    },
// JSON-NEXT:                    "hadMultipleCandidates": true,
// JSON-NEXT:                    "constructionKind": "complete"
// JSON-NEXT:                   }
// JSON-NEXT:                  ]
// JSON-NEXT:                 }
// JSON-NEXT:                ]
// JSON-NEXT:               }
// JSON-NEXT:              ]
// JSON-NEXT:             }
// JSON-NEXT:            ]
// JSON-NEXT:           }
// JSON-NEXT:          ]
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "DeclStmt",
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 1014,
// JSON-NEXT:          "line": 29,
// JSON-NEXT:          "col": 3,
// JSON-NEXT:          "tokLen": 6
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 1053,
// JSON-NEXT:          "col": 42,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "VarDecl",
// JSON-NEXT:          "loc": {
// JSON-NEXT:           "offset": 1021,
// JSON-NEXT:           "col": 10,
// JSON-NEXT:           "tokLen": 1
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 1014,
// JSON-NEXT:            "col": 3,
// JSON-NEXT:            "tokLen": 6
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 1052,
// JSON-NEXT:            "col": 41,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "name": "z",
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "double"
// JSON-NEXT:          },
// JSON-NEXT:          "init": "c",
// JSON-NEXT:          "inner": [
// JSON-NEXT:           {
// JSON-NEXT:            "id": "0x{{.*}}",
// JSON-NEXT:            "kind": "ExprWithCleanups",
// JSON-NEXT:            "range": {
// JSON-NEXT:             "begin": {
// JSON-NEXT:              "offset": 1025,
// JSON-NEXT:              "col": 14,
// JSON-NEXT:              "tokLen": 3
// JSON-NEXT:             },
// JSON-NEXT:             "end": {
// JSON-NEXT:              "offset": 1052,
// JSON-NEXT:              "col": 41,
// JSON-NEXT:              "tokLen": 1
// JSON-NEXT:             }
// JSON-NEXT:            },
// JSON-NEXT:            "type": {
// JSON-NEXT:             "qualType": "double"
// JSON-NEXT:            },
// JSON-NEXT:            "valueCategory": "prvalue",
// JSON-NEXT:            "inner": [
// JSON-NEXT:             {
// JSON-NEXT:              "id": "0x{{.*}}",
// JSON-NEXT:              "kind": "CXXMemberCallExpr",
// JSON-NEXT:              "range": {
// JSON-NEXT:               "begin": {
// JSON-NEXT:                "offset": 1025,
// JSON-NEXT:                "col": 14,
// JSON-NEXT:                "tokLen": 3
// JSON-NEXT:               },
// JSON-NEXT:               "end": {
// JSON-NEXT:                "offset": 1052,
// JSON-NEXT:                "col": 41,
// JSON-NEXT:                "tokLen": 1
// JSON-NEXT:               }
// JSON-NEXT:              },
// JSON-NEXT:              "type": {
// JSON-NEXT:               "qualType": "double"
// JSON-NEXT:              },
// JSON-NEXT:              "valueCategory": "prvalue",
// JSON-NEXT:              "inner": [
// JSON-NEXT:               {
// JSON-NEXT:                "id": "0x{{.*}}",
// JSON-NEXT:                "kind": "MemberExpr",
// JSON-NEXT:                "range": {
// JSON-NEXT:                 "begin": {
// JSON-NEXT:                  "offset": 1025,
// JSON-NEXT:                  "col": 14,
// JSON-NEXT:                  "tokLen": 3
// JSON-NEXT:                 },
// JSON-NEXT:                 "end": {
// JSON-NEXT:                  "offset": 1045,
// JSON-NEXT:                  "col": 34,
// JSON-NEXT:                  "tokLen": 6
// JSON-NEXT:                 }
// JSON-NEXT:                },
// JSON-NEXT:                "type": {
// JSON-NEXT:                 "qualType": "<bound member function type>"
// JSON-NEXT:                },
// JSON-NEXT:                "valueCategory": "prvalue",
// JSON-NEXT:                "name": "getSum",
// JSON-NEXT:                "isArrow": false,
// JSON-NEXT:                "referencedMemberDecl": "0x{{.*}}",
// JSON-NEXT:                "inner": [
// JSON-NEXT:                 {
// JSON-NEXT:                  "id": "0x{{.*}}",
// JSON-NEXT:                  "kind": "MaterializeTemporaryExpr",
// JSON-NEXT:                  "range": {
// JSON-NEXT:                   "begin": {
// JSON-NEXT:                    "offset": 1025,
// JSON-NEXT:                    "col": 14,
// JSON-NEXT:                    "tokLen": 3
// JSON-NEXT:                   },
// JSON-NEXT:                   "end": {
// JSON-NEXT:                    "offset": 1043,
// JSON-NEXT:                    "col": 32,
// JSON-NEXT:                    "tokLen": 1
// JSON-NEXT:                   }
// JSON-NEXT:                  },
// JSON-NEXT:                  "type": {
// JSON-NEXT:                   "qualType": "foo<2, double, 3>"
// JSON-NEXT:                  },
// JSON-NEXT:                  "valueCategory": "xvalue",
// JSON-NEXT:                  "storageDuration": "full expression",
// JSON-NEXT:                  "inner": [
// JSON-NEXT:                   {
// JSON-NEXT:                    "id": "0x{{.*}}",
// JSON-NEXT:                    "kind": "CXXTemporaryObjectExpr",
// JSON-NEXT:                    "range": {
// JSON-NEXT:                     "begin": {
// JSON-NEXT:                      "offset": 1025,
// JSON-NEXT:                      "col": 14,
// JSON-NEXT:                      "tokLen": 3
// JSON-NEXT:                     },
// JSON-NEXT:                     "end": {
// JSON-NEXT:                      "offset": 1043,
// JSON-NEXT:                      "col": 32,
// JSON-NEXT:                      "tokLen": 1
// JSON-NEXT:                     }
// JSON-NEXT:                    },
// JSON-NEXT:                    "type": {
// JSON-NEXT:                     "qualType": "foo<2, double, 3>"
// JSON-NEXT:                    },
// JSON-NEXT:                    "valueCategory": "prvalue",
// JSON-NEXT:                    "ctorType": {
// JSON-NEXT:                     "qualType": "void ()"
// JSON-NEXT:                    },
// JSON-NEXT:                    "hadMultipleCandidates": true,
// JSON-NEXT:                    "constructionKind": "complete"
// JSON-NEXT:                   }
// JSON-NEXT:                  ]
// JSON-NEXT:                 }
// JSON-NEXT:                ]
// JSON-NEXT:               }
// JSON-NEXT:              ]
// JSON-NEXT:             }
// JSON-NEXT:            ]
// JSON-NEXT:           }
// JSON-NEXT:          ]
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       }
// JSON-NEXT:      ]
// JSON-NEXT:     }
// JSON-NEXT:    ]
// JSON-NEXT:   },
// JSON-NEXT:   {
// JSON-NEXT:    "id": "0x{{.*}}",
// JSON-NEXT:    "kind": "ClassTemplateDecl",
// JSON-NEXT:    "loc": {
// JSON-NEXT:     "offset": 1856,
// JSON-NEXT:     "line": 52,
// JSON-NEXT:     "col": 33,
// JSON-NEXT:     "tokLen": 1
// JSON-NEXT:    },
// JSON-NEXT:    "range": {
// JSON-NEXT:     "begin": {
// JSON-NEXT:      "offset": 1824,
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 8
// JSON-NEXT:     },
// JSON-NEXT:     "end": {
// JSON-NEXT:      "offset": 1896,
// JSON-NEXT:      "line": 54,
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 1
// JSON-NEXT:     }
// JSON-NEXT:    },
// JSON-NEXT:    "name": "A",
// JSON-NEXT:    "inner": [
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "TemplateTypeParmDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": 1846,
// JSON-NEXT:       "line": 52,
// JSON-NEXT:       "col": 23,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 1834,
// JSON-NEXT:        "col": 11,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 1846,
// JSON-NEXT:        "col": 23,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "isReferenced": true,
// JSON-NEXT:      "name": "T",
// JSON-NEXT:      "tagUsed": "typename",
// JSON-NEXT:      "depth": 0,
// JSON-NEXT:      "index": 0,
// JSON-NEXT:      "isParameterPack": true
// JSON-NEXT:     },
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "CXXRecordDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": 1856,
// JSON-NEXT:       "col": 33,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 1849,
// JSON-NEXT:        "col": 26,
// JSON-NEXT:        "tokLen": 6
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 1896,
// JSON-NEXT:        "line": 54,
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "name": "A",
// JSON-NEXT:      "tagUsed": "struct",
// JSON-NEXT:      "completeDefinition": true,
// JSON-NEXT:      "definitionData": {
// JSON-NEXT:       "canConstDefaultInit": true,
// JSON-NEXT:       "copyAssign": {
// JSON-NEXT:        "hasConstParam": true,
// JSON-NEXT:        "implicitHasConstParam": true,
// JSON-NEXT:        "needsImplicit": true,
// JSON-NEXT:        "simple": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       },
// JSON-NEXT:       "copyCtor": {
// JSON-NEXT:        "hasConstParam": true,
// JSON-NEXT:        "implicitHasConstParam": true,
// JSON-NEXT:        "needsImplicit": true,
// JSON-NEXT:        "simple": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       },
// JSON-NEXT:       "defaultCtor": {
// JSON-NEXT:        "defaultedIsConstexpr": true,
// JSON-NEXT:        "exists": true,
// JSON-NEXT:        "isConstexpr": true,
// JSON-NEXT:        "needsImplicit": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       },
// JSON-NEXT:       "dtor": {
// JSON-NEXT:        "irrelevant": true,
// JSON-NEXT:        "needsImplicit": true,
// JSON-NEXT:        "simple": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       },
// JSON-NEXT:       "hasConstexprNonCopyMoveConstructor": true,
// JSON-NEXT:       "isAggregate": true,
// JSON-NEXT:       "isEmpty": true,
// JSON-NEXT:       "isLiteral": true,
// JSON-NEXT:       "isPOD": true,
// JSON-NEXT:       "isStandardLayout": true,
// JSON-NEXT:       "isTrivial": true,
// JSON-NEXT:       "isTriviallyCopyable": true,
// JSON-NEXT:       "moveAssign": {
// JSON-NEXT:        "exists": true,
// JSON-NEXT:        "needsImplicit": true,
// JSON-NEXT:        "simple": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       },
// JSON-NEXT:       "moveCtor": {
// JSON-NEXT:        "exists": true,
// JSON-NEXT:        "needsImplicit": true,
// JSON-NEXT:        "simple": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "inner": [
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "CXXRecordDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 1856,
// JSON-NEXT:         "line": 52,
// JSON-NEXT:         "col": 33,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 1849,
// JSON-NEXT:          "col": 26,
// JSON-NEXT:          "tokLen": 6
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 1856,
// JSON-NEXT:          "col": 33,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "isImplicit": true,
// JSON-NEXT:        "name": "A",
// JSON-NEXT:        "tagUsed": "struct"
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "ClassTemplateDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 1890,
// JSON-NEXT:         "line": 53,
// JSON-NEXT:         "col": 31,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 1862,
// JSON-NEXT:          "col": 3,
// JSON-NEXT:          "tokLen": 8
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 1893,
// JSON-NEXT:          "col": 34,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "name": "B",
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "NonTypeTemplateParmDecl",
// JSON-NEXT:          "loc": {
// JSON-NEXT:           "offset": 1877,
// JSON-NEXT:           "col": 18,
// JSON-NEXT:           "tokLen": 1
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 1872,
// JSON-NEXT:            "col": 13,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 1880,
// JSON-NEXT:            "col": 21,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "name": "x",
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "T[3]..."
// JSON-NEXT:          },
// JSON-NEXT:          "depth": 1,
// JSON-NEXT:          "index": 0,
// JSON-NEXT:          "isParameterPack": true
// JSON-NEXT:         },
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "CXXRecordDecl",
// JSON-NEXT:          "loc": {
// JSON-NEXT:           "offset": 1890,
// JSON-NEXT:           "col": 31,
// JSON-NEXT:           "tokLen": 1
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 1883,
// JSON-NEXT:            "col": 24,
// JSON-NEXT:            "tokLen": 6
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 1893,
// JSON-NEXT:            "col": 34,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "name": "B",
// JSON-NEXT:          "tagUsed": "struct",
// JSON-NEXT:          "completeDefinition": true,
// JSON-NEXT:          "definitionData": {
// JSON-NEXT:           "canConstDefaultInit": true,
// JSON-NEXT:           "copyAssign": {
// JSON-NEXT:            "hasConstParam": true,
// JSON-NEXT:            "implicitHasConstParam": true,
// JSON-NEXT:            "needsImplicit": true,
// JSON-NEXT:            "simple": true,
// JSON-NEXT:            "trivial": true
// JSON-NEXT:           },
// JSON-NEXT:           "copyCtor": {
// JSON-NEXT:            "hasConstParam": true,
// JSON-NEXT:            "implicitHasConstParam": true,
// JSON-NEXT:            "needsImplicit": true,
// JSON-NEXT:            "simple": true,
// JSON-NEXT:            "trivial": true
// JSON-NEXT:           },
// JSON-NEXT:           "defaultCtor": {
// JSON-NEXT:            "defaultedIsConstexpr": true,
// JSON-NEXT:            "exists": true,
// JSON-NEXT:            "isConstexpr": true,
// JSON-NEXT:            "needsImplicit": true,
// JSON-NEXT:            "trivial": true
// JSON-NEXT:           },
// JSON-NEXT:           "dtor": {
// JSON-NEXT:            "irrelevant": true,
// JSON-NEXT:            "needsImplicit": true,
// JSON-NEXT:            "simple": true,
// JSON-NEXT:            "trivial": true
// JSON-NEXT:           },
// JSON-NEXT:           "hasConstexprNonCopyMoveConstructor": true,
// JSON-NEXT:           "isAggregate": true,
// JSON-NEXT:           "isEmpty": true,
// JSON-NEXT:           "isLiteral": true,
// JSON-NEXT:           "isPOD": true,
// JSON-NEXT:           "isStandardLayout": true,
// JSON-NEXT:           "isTrivial": true,
// JSON-NEXT:           "isTriviallyCopyable": true,
// JSON-NEXT:           "moveAssign": {
// JSON-NEXT:            "exists": true,
// JSON-NEXT:            "needsImplicit": true,
// JSON-NEXT:            "simple": true,
// JSON-NEXT:            "trivial": true
// JSON-NEXT:           },
// JSON-NEXT:           "moveCtor": {
// JSON-NEXT:            "exists": true,
// JSON-NEXT:            "needsImplicit": true,
// JSON-NEXT:            "simple": true,
// JSON-NEXT:            "trivial": true
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "inner": [
// JSON-NEXT:           {
// JSON-NEXT:            "id": "0x{{.*}}",
// JSON-NEXT:            "kind": "CXXRecordDecl",
// JSON-NEXT:            "loc": {
// JSON-NEXT:             "offset": 1890,
// JSON-NEXT:             "col": 31,
// JSON-NEXT:             "tokLen": 1
// JSON-NEXT:            },
// JSON-NEXT:            "range": {
// JSON-NEXT:             "begin": {
// JSON-NEXT:              "offset": 1883,
// JSON-NEXT:              "col": 24,
// JSON-NEXT:              "tokLen": 6
// JSON-NEXT:             },
// JSON-NEXT:             "end": {
// JSON-NEXT:              "offset": 1890,
// JSON-NEXT:              "col": 31,
// JSON-NEXT:              "tokLen": 1
// JSON-NEXT:             }
// JSON-NEXT:            },
// JSON-NEXT:            "isImplicit": true,
// JSON-NEXT:            "name": "B",
// JSON-NEXT:            "tagUsed": "struct"
// JSON-NEXT:           }
// JSON-NEXT:          ]
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       }
// JSON-NEXT:      ]
// JSON-NEXT:     }
// JSON-NEXT:    ]
// JSON-NEXT:   },
// JSON-NEXT:   {
// JSON-NEXT:    "id": "0x{{.*}}",
// JSON-NEXT:    "kind": "FunctionTemplateDecl",
// JSON-NEXT:    "loc": {
// JSON-NEXT:     "offset": 2016,
// JSON-NEXT:     "line": 58,
// JSON-NEXT:     "col": 31,
// JSON-NEXT:     "tokLen": 1
// JSON-NEXT:    },
// JSON-NEXT:    "range": {
// JSON-NEXT:     "begin": {
// JSON-NEXT:      "offset": 1986,
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 8
// JSON-NEXT:     },
// JSON-NEXT:     "end": {
// JSON-NEXT:      "offset": 2038,
// JSON-NEXT:      "line": 60,
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 1
// JSON-NEXT:     }
// JSON-NEXT:    },
// JSON-NEXT:    "name": "f",
// JSON-NEXT:    "inner": [
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "TemplateTypeParmDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": 2008,
// JSON-NEXT:       "line": 58,
// JSON-NEXT:       "col": 23,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 1996,
// JSON-NEXT:        "col": 11,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 2008,
// JSON-NEXT:        "col": 23,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "isReferenced": true,
// JSON-NEXT:      "name": "T",
// JSON-NEXT:      "tagUsed": "typename",
// JSON-NEXT:      "depth": 0,
// JSON-NEXT:      "index": 0,
// JSON-NEXT:      "isParameterPack": true
// JSON-NEXT:     },
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "FunctionDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": 2016,
// JSON-NEXT:       "col": 31,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 2011,
// JSON-NEXT:        "col": 26,
// JSON-NEXT:        "tokLen": 4
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 2038,
// JSON-NEXT:        "line": 60,
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "name": "f",
// JSON-NEXT:      "type": {
// JSON-NEXT:       "qualType": "void ()"
// JSON-NEXT:      },
// JSON-NEXT:      "inner": [
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "CompoundStmt",
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 2020,
// JSON-NEXT:          "line": 58,
// JSON-NEXT:          "col": 35,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 2038,
// JSON-NEXT:          "line": 60,
// JSON-NEXT:          "col": 1,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "DeclStmt",
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 2024,
// JSON-NEXT:            "line": 59,
// JSON-NEXT:            "col": 3,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 2036,
// JSON-NEXT:            "col": 15,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "inner": [
// JSON-NEXT:           {
// JSON-NEXT:            "id": "0x{{.*}}",
// JSON-NEXT:            "kind": "VarDecl",
// JSON-NEXT:            "loc": {
// JSON-NEXT:             "offset": 2035,
// JSON-NEXT:             "col": 14,
// JSON-NEXT:             "tokLen": 1
// JSON-NEXT:            },
// JSON-NEXT:            "range": {
// JSON-NEXT:             "begin": {
// JSON-NEXT:              "offset": 2024,
// JSON-NEXT:              "col": 3,
// JSON-NEXT:              "tokLen": 1
// JSON-NEXT:             },
// JSON-NEXT:             "end": {
// JSON-NEXT:              "offset": 2035,
// JSON-NEXT:              "col": 14,
// JSON-NEXT:              "tokLen": 1
// JSON-NEXT:             }
// JSON-NEXT:            },
// JSON-NEXT:            "name": "a",
// JSON-NEXT:            "type": {
// JSON-NEXT:             "qualType": "A<T[3]...>"
// JSON-NEXT:            }
// JSON-NEXT:           }
// JSON-NEXT:          ]
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       }
// JSON-NEXT:      ]
// JSON-NEXT:     }
// JSON-NEXT:    ]
// JSON-NEXT:   },
// JSON-NEXT:   {
// JSON-NEXT:    "id": "0x{{.*}}",
// JSON-NEXT:    "kind": "NamespaceDecl",
// JSON-NEXT:    "loc": {
// JSON-NEXT:     "offset": 2051,
// JSON-NEXT:     "line": 62,
// JSON-NEXT:     "col": 11,
// JSON-NEXT:     "tokLen": 5
// JSON-NEXT:    },
// JSON-NEXT:    "range": {
// JSON-NEXT:     "begin": {
// JSON-NEXT:      "offset": 2041,
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 9
// JSON-NEXT:     },
// JSON-NEXT:     "end": {
// JSON-NEXT:      "offset": 2240,
// JSON-NEXT:      "line": 71,
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 1
// JSON-NEXT:     }
// JSON-NEXT:    },
// JSON-NEXT:    "name": "test2",
// JSON-NEXT:    "inner": [
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "FunctionDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": 2064,
// JSON-NEXT:       "line": 63,
// JSON-NEXT:       "col": 6,
// JSON-NEXT:       "tokLen": 4
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 2059,
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 4
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 2072,
// JSON-NEXT:        "col": 14,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "name": "func",
// JSON-NEXT:      "mangledName": "_ZN5test24funcEi",
// JSON-NEXT:      "type": {
// JSON-NEXT:       "qualType": "void (int)"
// JSON-NEXT:      },
// JSON-NEXT:      "inner": [
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "ParmVarDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 2072,
// JSON-NEXT:         "col": 14,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 2069,
// JSON-NEXT:          "col": 11,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 2069,
// JSON-NEXT:          "col": 11,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "int"
// JSON-NEXT:        }
// JSON-NEXT:       }
// JSON-NEXT:      ]
// JSON-NEXT:     },
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "FunctionDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": 2080,
// JSON-NEXT:       "line": 64,
// JSON-NEXT:       "col": 6,
// JSON-NEXT:       "tokLen": 4
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 2075,
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 4
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 2090,
// JSON-NEXT:        "col": 16,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "name": "func",
// JSON-NEXT:      "mangledName": "_ZN5test24funcEf",
// JSON-NEXT:      "type": {
// JSON-NEXT:       "qualType": "void (float)"
// JSON-NEXT:      },
// JSON-NEXT:      "inner": [
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "ParmVarDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 2090,
// JSON-NEXT:         "col": 16,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 2085,
// JSON-NEXT:          "col": 11,
// JSON-NEXT:          "tokLen": 5
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 2085,
// JSON-NEXT:          "col": 11,
// JSON-NEXT:          "tokLen": 5
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "float"
// JSON-NEXT:        }
// JSON-NEXT:       }
// JSON-NEXT:      ]
// JSON-NEXT:     },
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "FunctionTemplateDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": 2119,
// JSON-NEXT:       "line": 66,
// JSON-NEXT:       "col": 6,
// JSON-NEXT:       "tokLen": 4
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 2093,
// JSON-NEXT:        "line": 65,
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 2141,
// JSON-NEXT:        "line": 68,
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "name": "tmpl",
// JSON-NEXT:      "inner": [
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "TemplateTypeParmDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 2111,
// JSON-NEXT:         "line": 65,
// JSON-NEXT:         "col": 19,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 2102,
// JSON-NEXT:          "col": 10,
// JSON-NEXT:          "tokLen": 8
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 2111,
// JSON-NEXT:          "col": 19,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "isReferenced": true,
// JSON-NEXT:        "name": "T",
// JSON-NEXT:        "tagUsed": "typename",
// JSON-NEXT:        "depth": 0,
// JSON-NEXT:        "index": 0
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "FunctionDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 2119,
// JSON-NEXT:         "line": 66,
// JSON-NEXT:         "col": 6,
// JSON-NEXT:         "tokLen": 4
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 2114,
// JSON-NEXT:          "col": 1,
// JSON-NEXT:          "tokLen": 4
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 2141,
// JSON-NEXT:          "line": 68,
// JSON-NEXT:          "col": 1,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "name": "tmpl",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "void ()"
// JSON-NEXT:        },
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "CompoundStmt",
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 2126,
// JSON-NEXT:            "line": 66,
// JSON-NEXT:            "col": 13,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 2141,
// JSON-NEXT:            "line": 68,
// JSON-NEXT:            "col": 1,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "inner": [
// JSON-NEXT:           {
// JSON-NEXT:            "id": "0x{{.*}}",
// JSON-NEXT:            "kind": "CallExpr",
// JSON-NEXT:            "range": {
// JSON-NEXT:             "begin": {
// JSON-NEXT:              "offset": 2130,
// JSON-NEXT:              "line": 67,
// JSON-NEXT:              "col": 3,
// JSON-NEXT:              "tokLen": 4
// JSON-NEXT:             },
// JSON-NEXT:             "end": {
// JSON-NEXT:              "offset": 2138,
// JSON-NEXT:              "col": 11,
// JSON-NEXT:              "tokLen": 1
// JSON-NEXT:             }
// JSON-NEXT:            },
// JSON-NEXT:            "type": {
// JSON-NEXT:             "qualType": "<dependent type>"
// JSON-NEXT:            },
// JSON-NEXT:            "valueCategory": "prvalue",
// JSON-NEXT:            "inner": [
// JSON-NEXT:             {
// JSON-NEXT:              "id": "0x{{.*}}",
// JSON-NEXT:              "kind": "UnresolvedLookupExpr",
// JSON-NEXT:              "range": {
// JSON-NEXT:               "begin": {
// JSON-NEXT:                "offset": 2130,
// JSON-NEXT:                "col": 3,
// JSON-NEXT:                "tokLen": 4
// JSON-NEXT:               },
// JSON-NEXT:               "end": {
// JSON-NEXT:                "offset": 2130,
// JSON-NEXT:                "col": 3,
// JSON-NEXT:                "tokLen": 4
// JSON-NEXT:               }
// JSON-NEXT:              },
// JSON-NEXT:              "type": {
// JSON-NEXT:               "qualType": "<overloaded function type>"
// JSON-NEXT:              },
// JSON-NEXT:              "valueCategory": "lvalue",
// JSON-NEXT:              "usesADL": true,
// JSON-NEXT:              "name": "func",
// JSON-NEXT:              "lookups": [
// JSON-NEXT:               {
// JSON-NEXT:                "id": "0x{{.*}}",
// JSON-NEXT:                "kind": "FunctionDecl",
// JSON-NEXT:                "name": "func",
// JSON-NEXT:                "type": {
// JSON-NEXT:                 "qualType": "void (float)"
// JSON-NEXT:                }
// JSON-NEXT:               },
// JSON-NEXT:               {
// JSON-NEXT:                "id": "0x{{.*}}",
// JSON-NEXT:                "kind": "FunctionDecl",
// JSON-NEXT:                "name": "func",
// JSON-NEXT:                "type": {
// JSON-NEXT:                 "qualType": "void (int)"
// JSON-NEXT:                }
// JSON-NEXT:               }
// JSON-NEXT:              ]
// JSON-NEXT:             },
// JSON-NEXT:             {
// JSON-NEXT:              "id": "0x{{.*}}",
// JSON-NEXT:              "kind": "CXXUnresolvedConstructExpr",
// JSON-NEXT:              "range": {
// JSON-NEXT:               "begin": {
// JSON-NEXT:                "offset": 2135,
// JSON-NEXT:                "col": 8,
// JSON-NEXT:                "tokLen": 1
// JSON-NEXT:               },
// JSON-NEXT:               "end": {
// JSON-NEXT:                "offset": 2137,
// JSON-NEXT:                "col": 10,
// JSON-NEXT:                "tokLen": 1
// JSON-NEXT:               }
// JSON-NEXT:              },
// JSON-NEXT:              "type": {
// JSON-NEXT:               "qualType": "T"
// JSON-NEXT:              },
// JSON-NEXT:              "valueCategory": "prvalue"
// JSON-NEXT:             }
// JSON-NEXT:            ]
// JSON-NEXT:           }
// JSON-NEXT:          ]
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       }
// JSON-NEXT:      ]
// JSON-NEXT:     }
// JSON-NEXT:    ]
// JSON-NEXT:   },
// JSON-NEXT:   {
// JSON-NEXT:    "id": "0x{{.*}}",
// JSON-NEXT:    "kind": "NamespaceDecl",
// JSON-NEXT:    "loc": {
// JSON-NEXT:     "offset": 2253,
// JSON-NEXT:     "line": 73,
// JSON-NEXT:     "col": 11,
// JSON-NEXT:     "tokLen": 5
// JSON-NEXT:    },
// JSON-NEXT:    "range": {
// JSON-NEXT:     "begin": {
// JSON-NEXT:      "offset": 2243,
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 9
// JSON-NEXT:     },
// JSON-NEXT:     "end": {
// JSON-NEXT:      "offset": 2387,
// JSON-NEXT:      "line": 77,
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 1
// JSON-NEXT:     }
// JSON-NEXT:    },
// JSON-NEXT:    "name": "test3",
// JSON-NEXT:    "inner": [
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "ClassTemplateDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": 2291,
// JSON-NEXT:       "line": 74,
// JSON-NEXT:       "col": 31,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 2263,
// JSON-NEXT:        "col": 3,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 2294,
// JSON-NEXT:        "col": 34,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "name": "A",
// JSON-NEXT:      "inner": [
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "TemplateTypeParmDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 2281,
// JSON-NEXT:         "col": 21,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 2272,
// JSON-NEXT:          "col": 12,
// JSON-NEXT:          "tokLen": 8
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 2281,
// JSON-NEXT:          "col": 21,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "name": "T",
// JSON-NEXT:        "tagUsed": "typename",
// JSON-NEXT:        "depth": 0,
// JSON-NEXT:        "index": 0
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "CXXRecordDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 2291,
// JSON-NEXT:         "col": 31,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 2284,
// JSON-NEXT:          "col": 24,
// JSON-NEXT:          "tokLen": 6
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 2294,
// JSON-NEXT:          "col": 34,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "name": "A",
// JSON-NEXT:        "tagUsed": "struct",
// JSON-NEXT:        "completeDefinition": true,
// JSON-NEXT:        "definitionData": {
// JSON-NEXT:         "canConstDefaultInit": true,
// JSON-NEXT:         "copyAssign": {
// JSON-NEXT:          "hasConstParam": true,
// JSON-NEXT:          "implicitHasConstParam": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "simple": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         },
// JSON-NEXT:         "copyCtor": {
// JSON-NEXT:          "hasConstParam": true,
// JSON-NEXT:          "implicitHasConstParam": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "simple": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         },
// JSON-NEXT:         "defaultCtor": {
// JSON-NEXT:          "defaultedIsConstexpr": true,
// JSON-NEXT:          "exists": true,
// JSON-NEXT:          "isConstexpr": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         },
// JSON-NEXT:         "dtor": {
// JSON-NEXT:          "irrelevant": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "simple": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         },
// JSON-NEXT:         "hasConstexprNonCopyMoveConstructor": true,
// JSON-NEXT:         "isAggregate": true,
// JSON-NEXT:         "isEmpty": true,
// JSON-NEXT:         "isLiteral": true,
// JSON-NEXT:         "isPOD": true,
// JSON-NEXT:         "isStandardLayout": true,
// JSON-NEXT:         "isTrivial": true,
// JSON-NEXT:         "isTriviallyCopyable": true,
// JSON-NEXT:         "moveAssign": {
// JSON-NEXT:          "exists": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "simple": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         },
// JSON-NEXT:         "moveCtor": {
// JSON-NEXT:          "exists": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "simple": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "CXXRecordDecl",
// JSON-NEXT:          "loc": {
// JSON-NEXT:           "offset": 2291,
// JSON-NEXT:           "col": 31,
// JSON-NEXT:           "tokLen": 1
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 2284,
// JSON-NEXT:            "col": 24,
// JSON-NEXT:            "tokLen": 6
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 2291,
// JSON-NEXT:            "col": 31,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "isImplicit": true,
// JSON-NEXT:          "name": "A",
// JSON-NEXT:          "tagUsed": "struct"
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "ClassTemplateSpecializationDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 2291,
// JSON-NEXT:         "col": 31,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 2263,
// JSON-NEXT:          "col": 3,
// JSON-NEXT:          "tokLen": 8
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 2294,
// JSON-NEXT:          "col": 34,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "name": "A",
// JSON-NEXT:        "tagUsed": "struct",
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "kind": "TemplateArgument",
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "int"
// JSON-NEXT:          },
// JSON-NEXT:          "inner": [
// JSON-NEXT:           {
// JSON-NEXT:            "id": "0x{{.*}}",
// JSON-NEXT:            "kind": "BuiltinType",
// JSON-NEXT:            "type": {
// JSON-NEXT:             "qualType": "int"
// JSON-NEXT:            }
// JSON-NEXT:           }
// JSON-NEXT:          ]
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       }
// JSON-NEXT:      ]
// JSON-NEXT:     },
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "FunctionTemplateDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": 2291,
// JSON-NEXT:       "col": 31,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 2263,
// JSON-NEXT:        "col": 3,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 2291,
// JSON-NEXT:        "col": 31,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "isImplicit": true,
// JSON-NEXT:      "name": "<deduction guide for A>",
// JSON-NEXT:      "inner": [
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "TemplateTypeParmDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 2281,
// JSON-NEXT:         "col": 21,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 2272,
// JSON-NEXT:          "col": 12,
// JSON-NEXT:          "tokLen": 8
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 2281,
// JSON-NEXT:          "col": 21,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "name": "T",
// JSON-NEXT:        "tagUsed": "typename",
// JSON-NEXT:        "depth": 0,
// JSON-NEXT:        "index": 0
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "CXXDeductionGuideDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 2291,
// JSON-NEXT:         "col": 31,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 2291,
// JSON-NEXT:          "col": 31,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 2291,
// JSON-NEXT:          "col": 31,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "isImplicit": true,
// JSON-NEXT:        "name": "<deduction guide for A>",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "auto () -> test3::A<T>"
// JSON-NEXT:        }
// JSON-NEXT:       }
// JSON-NEXT:      ]
// JSON-NEXT:     },
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "FunctionTemplateDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": 2291,
// JSON-NEXT:       "col": 31,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 2263,
// JSON-NEXT:        "col": 3,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 2291,
// JSON-NEXT:        "col": 31,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "isImplicit": true,
// JSON-NEXT:      "name": "<deduction guide for A>",
// JSON-NEXT:      "inner": [
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "TemplateTypeParmDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 2281,
// JSON-NEXT:         "col": 21,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 2272,
// JSON-NEXT:          "col": 12,
// JSON-NEXT:          "tokLen": 8
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 2281,
// JSON-NEXT:          "col": 21,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "name": "T",
// JSON-NEXT:        "tagUsed": "typename",
// JSON-NEXT:        "depth": 0,
// JSON-NEXT:        "index": 0
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "CXXDeductionGuideDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 2291,
// JSON-NEXT:         "col": 31,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 2291,
// JSON-NEXT:          "col": 31,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 2291,
// JSON-NEXT:          "col": 31,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "isImplicit": true,
// JSON-NEXT:        "name": "<deduction guide for A>",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "auto (test3::A<T>) -> test3::A<T>"
// JSON-NEXT:        },
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "ParmVarDecl",
// JSON-NEXT:          "loc": {
// JSON-NEXT:           "offset": 2291,
// JSON-NEXT:           "col": 31,
// JSON-NEXT:           "tokLen": 1
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 2291,
// JSON-NEXT:            "col": 31,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 2291,
// JSON-NEXT:            "col": 31,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "test3::A<T>"
// JSON-NEXT:          }
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       }
// JSON-NEXT:      ]
// JSON-NEXT:     },
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "FunctionTemplateDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": 2320,
// JSON-NEXT:       "line": 75,
// JSON-NEXT:       "col": 24,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 2299,
// JSON-NEXT:        "col": 3,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 2333,
// JSON-NEXT:        "col": 37,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "name": "<deduction guide for A>",
// JSON-NEXT:      "inner": [
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "TemplateTypeParmDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 2317,
// JSON-NEXT:         "col": 21,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 2308,
// JSON-NEXT:          "col": 12,
// JSON-NEXT:          "tokLen": 8
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 2317,
// JSON-NEXT:          "col": 21,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "isReferenced": true,
// JSON-NEXT:        "name": "T",
// JSON-NEXT:        "tagUsed": "typename",
// JSON-NEXT:        "depth": 0,
// JSON-NEXT:        "index": 0
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "CXXDeductionGuideDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 2320,
// JSON-NEXT:         "col": 24,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 2320,
// JSON-NEXT:          "col": 24,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 2333,
// JSON-NEXT:          "col": 37,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "name": "<deduction guide for A>",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "auto (T) -> A<int>"
// JSON-NEXT:        },
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "ParmVarDecl",
// JSON-NEXT:          "loc": {
// JSON-NEXT:           "offset": 2323,
// JSON-NEXT:           "col": 27,
// JSON-NEXT:           "tokLen": 1
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 2322,
// JSON-NEXT:            "col": 26,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 2322,
// JSON-NEXT:            "col": 26,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "T"
// JSON-NEXT:          }
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       }
// JSON-NEXT:      ]
// JSON-NEXT:     }
// JSON-NEXT:    ]
// JSON-NEXT:   },
// JSON-NEXT:   {
// JSON-NEXT:    "id": "0x{{.*}}",
// JSON-NEXT:    "kind": "NamespaceDecl",
// JSON-NEXT:    "loc": {
// JSON-NEXT:     "offset": 2400,
// JSON-NEXT:     "line": 79,
// JSON-NEXT:     "col": 11,
// JSON-NEXT:     "tokLen": 5
// JSON-NEXT:    },
// JSON-NEXT:    "range": {
// JSON-NEXT:     "begin": {
// JSON-NEXT:      "offset": 2390,
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 9
// JSON-NEXT:     },
// JSON-NEXT:     "end": {
// JSON-NEXT:      "offset": 3297,
// JSON-NEXT:      "line": 103,
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 1
// JSON-NEXT:     }
// JSON-NEXT:    },
// JSON-NEXT:    "name": "test4",
// JSON-NEXT:    "inner": [
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "ClassTemplateDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": 2445,
// JSON-NEXT:       "line": 81,
// JSON-NEXT:       "col": 8,
// JSON-NEXT:       "tokLen": 3
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 2408,
// JSON-NEXT:        "line": 80,
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 2471,
// JSON-NEXT:        "line": 83,
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "name": "foo",
// JSON-NEXT:      "inner": [
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "NonTypeTemplateParmDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 2427,
// JSON-NEXT:         "line": 80,
// JSON-NEXT:         "col": 20,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 2418,
// JSON-NEXT:          "col": 11,
// JSON-NEXT:          "tokLen": 8
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 2427,
// JSON-NEXT:          "col": 20,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "name": "X",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "unsigned int"
// JSON-NEXT:        },
// JSON-NEXT:        "depth": 0,
// JSON-NEXT:        "index": 0
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "NonTypeTemplateParmDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 2435,
// JSON-NEXT:         "col": 28,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 2430,
// JSON-NEXT:          "col": 23,
// JSON-NEXT:          "tokLen": 4
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 2435,
// JSON-NEXT:          "col": 28,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "name": "A",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "auto"
// JSON-NEXT:        },
// JSON-NEXT:        "depth": 0,
// JSON-NEXT:        "index": 1
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "CXXRecordDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 2445,
// JSON-NEXT:         "line": 81,
// JSON-NEXT:         "col": 8,
// JSON-NEXT:         "tokLen": 3
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 2438,
// JSON-NEXT:          "col": 1,
// JSON-NEXT:          "tokLen": 6
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 2471,
// JSON-NEXT:          "line": 83,
// JSON-NEXT:          "col": 1,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "name": "foo",
// JSON-NEXT:        "tagUsed": "struct",
// JSON-NEXT:        "completeDefinition": true,
// JSON-NEXT:        "definitionData": {
// JSON-NEXT:         "canConstDefaultInit": true,
// JSON-NEXT:         "copyAssign": {
// JSON-NEXT:          "hasConstParam": true,
// JSON-NEXT:          "implicitHasConstParam": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "simple": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         },
// JSON-NEXT:         "copyCtor": {
// JSON-NEXT:          "hasConstParam": true,
// JSON-NEXT:          "implicitHasConstParam": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "simple": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         },
// JSON-NEXT:         "defaultCtor": {
// JSON-NEXT:          "defaultedIsConstexpr": true,
// JSON-NEXT:          "exists": true,
// JSON-NEXT:          "isConstexpr": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         },
// JSON-NEXT:         "dtor": {
// JSON-NEXT:          "irrelevant": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "simple": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         },
// JSON-NEXT:         "hasConstexprNonCopyMoveConstructor": true,
// JSON-NEXT:         "isAggregate": true,
// JSON-NEXT:         "isEmpty": true,
// JSON-NEXT:         "isLiteral": true,
// JSON-NEXT:         "isPOD": true,
// JSON-NEXT:         "isStandardLayout": true,
// JSON-NEXT:         "isTrivial": true,
// JSON-NEXT:         "isTriviallyCopyable": true,
// JSON-NEXT:         "moveAssign": {
// JSON-NEXT:          "exists": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "simple": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         },
// JSON-NEXT:         "moveCtor": {
// JSON-NEXT:          "exists": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "simple": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "CXXRecordDecl",
// JSON-NEXT:          "loc": {
// JSON-NEXT:           "offset": 2445,
// JSON-NEXT:           "line": 81,
// JSON-NEXT:           "col": 8,
// JSON-NEXT:           "tokLen": 3
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 2438,
// JSON-NEXT:            "col": 1,
// JSON-NEXT:            "tokLen": 6
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 2445,
// JSON-NEXT:            "col": 8,
// JSON-NEXT:            "tokLen": 3
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "isImplicit": true,
// JSON-NEXT:          "name": "foo",
// JSON-NEXT:          "tagUsed": "struct"
// JSON-NEXT:         },
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "CXXMethodDecl",
// JSON-NEXT:          "loc": {
// JSON-NEXT:           "offset": 2465,
// JSON-NEXT:           "line": 82,
// JSON-NEXT:           "col": 15,
// JSON-NEXT:           "tokLen": 2
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 2453,
// JSON-NEXT:            "col": 3,
// JSON-NEXT:            "tokLen": 6
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 2468,
// JSON-NEXT:            "col": 18,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "name": "fn",
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "void ()"
// JSON-NEXT:          },
// JSON-NEXT:          "storageClass": "static"
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "ClassTemplateSpecializationDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 2445,
// JSON-NEXT:         "line": 81,
// JSON-NEXT:         "col": 8,
// JSON-NEXT:         "tokLen": 3
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 2408,
// JSON-NEXT:          "line": 80,
// JSON-NEXT:          "col": 1,
// JSON-NEXT:          "tokLen": 8
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 2471,
// JSON-NEXT:          "line": 83,
// JSON-NEXT:          "col": 1,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "name": "foo",
// JSON-NEXT:        "tagUsed": "struct",
// JSON-NEXT:        "completeDefinition": true,
// JSON-NEXT:        "definitionData": {
// JSON-NEXT:         "canConstDefaultInit": true,
// JSON-NEXT:         "canPassInRegisters": true,
// JSON-NEXT:         "copyAssign": {
// JSON-NEXT:          "hasConstParam": true,
// JSON-NEXT:          "implicitHasConstParam": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "simple": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         },
// JSON-NEXT:         "copyCtor": {
// JSON-NEXT:          "hasConstParam": true,
// JSON-NEXT:          "implicitHasConstParam": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "simple": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         },
// JSON-NEXT:         "defaultCtor": {
// JSON-NEXT:          "defaultedIsConstexpr": true,
// JSON-NEXT:          "exists": true,
// JSON-NEXT:          "isConstexpr": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         },
// JSON-NEXT:         "dtor": {
// JSON-NEXT:          "irrelevant": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "simple": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         },
// JSON-NEXT:         "hasConstexprNonCopyMoveConstructor": true,
// JSON-NEXT:         "isAggregate": true,
// JSON-NEXT:         "isEmpty": true,
// JSON-NEXT:         "isLiteral": true,
// JSON-NEXT:         "isPOD": true,
// JSON-NEXT:         "isStandardLayout": true,
// JSON-NEXT:         "isTrivial": true,
// JSON-NEXT:         "isTriviallyCopyable": true,
// JSON-NEXT:         "moveAssign": {
// JSON-NEXT:          "exists": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "simple": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         },
// JSON-NEXT:         "moveCtor": {
// JSON-NEXT:          "exists": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "simple": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "kind": "TemplateArgument",
// JSON-NEXT:          "value": 0
// JSON-NEXT:         },
// JSON-NEXT:         {
// JSON-NEXT:          "kind": "TemplateArgument",
// JSON-NEXT:          "value": 0
// JSON-NEXT:         },
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "CXXRecordDecl",
// JSON-NEXT:          "loc": {
// JSON-NEXT:           "offset": 2445,
// JSON-NEXT:           "line": 81,
// JSON-NEXT:           "col": 8,
// JSON-NEXT:           "tokLen": 3
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 2438,
// JSON-NEXT:            "col": 1,
// JSON-NEXT:            "tokLen": 6
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 2445,
// JSON-NEXT:            "col": 8,
// JSON-NEXT:            "tokLen": 3
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "isImplicit": true,
// JSON-NEXT:          "name": "foo",
// JSON-NEXT:          "tagUsed": "struct"
// JSON-NEXT:         },
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "CXXMethodDecl",
// JSON-NEXT:          "loc": {
// JSON-NEXT:           "offset": 2465,
// JSON-NEXT:           "line": 82,
// JSON-NEXT:           "col": 15,
// JSON-NEXT:           "tokLen": 2
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 2453,
// JSON-NEXT:            "col": 3,
// JSON-NEXT:            "tokLen": 6
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 2468,
// JSON-NEXT:            "col": 18,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "isUsed": true,
// JSON-NEXT:          "name": "fn",
// JSON-NEXT:          "mangledName": "_ZN5test43fooILj0ELl0EE2fnEv",
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "void ()"
// JSON-NEXT:          },
// JSON-NEXT:          "storageClass": "static"
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "ClassTemplateSpecializationDecl",
// JSON-NEXT:        "name": "foo"
// JSON-NEXT:       }
// JSON-NEXT:      ]
// JSON-NEXT:     },
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "FunctionDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": 2846,
// JSON-NEXT:       "line": 92,
// JSON-NEXT:       "col": 6,
// JSON-NEXT:       "tokLen": 4
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 2841,
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 4
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 2879,
// JSON-NEXT:        "line": 94,
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "name": "test",
// JSON-NEXT:      "mangledName": "_ZN5test44testEv",
// JSON-NEXT:      "type": {
// JSON-NEXT:       "qualType": "void ()"
// JSON-NEXT:      },
// JSON-NEXT:      "inner": [
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "CompoundStmt",
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 2853,
// JSON-NEXT:          "line": 92,
// JSON-NEXT:          "col": 13,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 2879,
// JSON-NEXT:          "line": 94,
// JSON-NEXT:          "col": 1,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "CallExpr",
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 2857,
// JSON-NEXT:            "line": 93,
// JSON-NEXT:            "col": 3,
// JSON-NEXT:            "tokLen": 3
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 2876,
// JSON-NEXT:            "col": 22,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "void"
// JSON-NEXT:          },
// JSON-NEXT:          "valueCategory": "prvalue",
// JSON-NEXT:          "inner": [
// JSON-NEXT:           {
// JSON-NEXT:            "id": "0x{{.*}}",
// JSON-NEXT:            "kind": "ImplicitCastExpr",
// JSON-NEXT:            "range": {
// JSON-NEXT:             "begin": {
// JSON-NEXT:              "offset": 2857,
// JSON-NEXT:              "col": 3,
// JSON-NEXT:              "tokLen": 3
// JSON-NEXT:             },
// JSON-NEXT:             "end": {
// JSON-NEXT:              "offset": 2873,
// JSON-NEXT:              "col": 19,
// JSON-NEXT:              "tokLen": 2
// JSON-NEXT:             }
// JSON-NEXT:            },
// JSON-NEXT:            "type": {
// JSON-NEXT:             "qualType": "void (*)()"
// JSON-NEXT:            },
// JSON-NEXT:            "valueCategory": "prvalue",
// JSON-NEXT:            "castKind": "FunctionToPointerDecay",
// JSON-NEXT:            "inner": [
// JSON-NEXT:             {
// JSON-NEXT:              "id": "0x{{.*}}",
// JSON-NEXT:              "kind": "DeclRefExpr",
// JSON-NEXT:              "range": {
// JSON-NEXT:               "begin": {
// JSON-NEXT:                "offset": 2857,
// JSON-NEXT:                "col": 3,
// JSON-NEXT:                "tokLen": 3
// JSON-NEXT:               },
// JSON-NEXT:               "end": {
// JSON-NEXT:                "offset": 2873,
// JSON-NEXT:                "col": 19,
// JSON-NEXT:                "tokLen": 2
// JSON-NEXT:               }
// JSON-NEXT:              },
// JSON-NEXT:              "type": {
// JSON-NEXT:               "qualType": "void ()"
// JSON-NEXT:              },
// JSON-NEXT:              "valueCategory": "lvalue",
// JSON-NEXT:              "referencedDecl": {
// JSON-NEXT:               "id": "0x{{.*}}",
// JSON-NEXT:               "kind": "CXXMethodDecl",
// JSON-NEXT:               "name": "fn",
// JSON-NEXT:               "type": {
// JSON-NEXT:                "qualType": "void ()"
// JSON-NEXT:               }
// JSON-NEXT:              }
// JSON-NEXT:             }
// JSON-NEXT:            ]
// JSON-NEXT:           }
// JSON-NEXT:          ]
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       }
// JSON-NEXT:      ]
// JSON-NEXT:     },
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "ClassTemplateSpecializationDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": 3281,
// JSON-NEXT:       "line": 102,
// JSON-NEXT:       "col": 17,
// JSON-NEXT:       "tokLen": 3
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 3265,
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 3294,
// JSON-NEXT:        "col": 30,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "name": "foo",
// JSON-NEXT:      "tagUsed": "struct",
// JSON-NEXT:      "completeDefinition": true,
// JSON-NEXT:      "definitionData": {
// JSON-NEXT:       "canConstDefaultInit": true,
// JSON-NEXT:       "canPassInRegisters": true,
// JSON-NEXT:       "copyAssign": {
// JSON-NEXT:        "hasConstParam": true,
// JSON-NEXT:        "implicitHasConstParam": true,
// JSON-NEXT:        "needsImplicit": true,
// JSON-NEXT:        "simple": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       },
// JSON-NEXT:       "copyCtor": {
// JSON-NEXT:        "hasConstParam": true,
// JSON-NEXT:        "implicitHasConstParam": true,
// JSON-NEXT:        "needsImplicit": true,
// JSON-NEXT:        "simple": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       },
// JSON-NEXT:       "defaultCtor": {
// JSON-NEXT:        "defaultedIsConstexpr": true,
// JSON-NEXT:        "exists": true,
// JSON-NEXT:        "isConstexpr": true,
// JSON-NEXT:        "needsImplicit": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       },
// JSON-NEXT:       "dtor": {
// JSON-NEXT:        "irrelevant": true,
// JSON-NEXT:        "needsImplicit": true,
// JSON-NEXT:        "simple": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       },
// JSON-NEXT:       "hasConstexprNonCopyMoveConstructor": true,
// JSON-NEXT:       "isAggregate": true,
// JSON-NEXT:       "isEmpty": true,
// JSON-NEXT:       "isLiteral": true,
// JSON-NEXT:       "isPOD": true,
// JSON-NEXT:       "isStandardLayout": true,
// JSON-NEXT:       "isTrivial": true,
// JSON-NEXT:       "isTriviallyCopyable": true,
// JSON-NEXT:       "moveAssign": {
// JSON-NEXT:        "exists": true,
// JSON-NEXT:        "needsImplicit": true,
// JSON-NEXT:        "simple": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       },
// JSON-NEXT:       "moveCtor": {
// JSON-NEXT:        "exists": true,
// JSON-NEXT:        "needsImplicit": true,
// JSON-NEXT:        "simple": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "inner": [
// JSON-NEXT:       {
// JSON-NEXT:        "kind": "TemplateArgument",
// JSON-NEXT:        "value": 1
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "kind": "TemplateArgument",
// JSON-NEXT:        "value": 0
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "CXXRecordDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 2445,
// JSON-NEXT:         "line": 81,
// JSON-NEXT:         "col": 8,
// JSON-NEXT:         "tokLen": 3
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 2438,
// JSON-NEXT:          "col": 1,
// JSON-NEXT:          "tokLen": 6
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 2445,
// JSON-NEXT:          "col": 8,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "isImplicit": true,
// JSON-NEXT:        "name": "foo",
// JSON-NEXT:        "tagUsed": "struct"
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "CXXMethodDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 2465,
// JSON-NEXT:         "line": 82,
// JSON-NEXT:         "col": 15,
// JSON-NEXT:         "tokLen": 2
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 2453,
// JSON-NEXT:          "col": 3,
// JSON-NEXT:          "tokLen": 6
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 2468,
// JSON-NEXT:          "col": 18,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "name": "fn",
// JSON-NEXT:        "mangledName": "_ZN5test43fooILj1ELl0EE2fnEv",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "void ()"
// JSON-NEXT:        },
// JSON-NEXT:        "storageClass": "static"
// JSON-NEXT:       }
// JSON-NEXT:      ]
// JSON-NEXT:     }
// JSON-NEXT:    ]
// JSON-NEXT:   },
// JSON-NEXT:   {
// JSON-NEXT:    "id": "0x{{.*}}",
// JSON-NEXT:    "kind": "NamespaceDecl",
// JSON-NEXT:    "loc": {
// JSON-NEXT:     "offset": 3310,
// JSON-NEXT:     "line": 105,
// JSON-NEXT:     "col": 11,
// JSON-NEXT:     "tokLen": 5
// JSON-NEXT:    },
// JSON-NEXT:    "range": {
// JSON-NEXT:     "begin": {
// JSON-NEXT:      "offset": 3300,
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 9
// JSON-NEXT:     },
// JSON-NEXT:     "end": {
// JSON-NEXT:      "offset": 3632,
// JSON-NEXT:      "line": 114,
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 1
// JSON-NEXT:     }
// JSON-NEXT:    },
// JSON-NEXT:    "name": "test5",
// JSON-NEXT:    "inner": [
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "FunctionTemplateDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": 3338,
// JSON-NEXT:       "line": 106,
// JSON-NEXT:       "col": 21,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 3318,
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 3343,
// JSON-NEXT:        "col": 26,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "name": "f",
// JSON-NEXT:      "inner": [
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "NonTypeTemplateParmDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 3331,
// JSON-NEXT:         "col": 14,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 3327,
// JSON-NEXT:          "col": 10,
// JSON-NEXT:          "tokLen": 4
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 3327,
// JSON-NEXT:          "col": 10,
// JSON-NEXT:          "tokLen": 4
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "long"
// JSON-NEXT:        },
// JSON-NEXT:        "depth": 0,
// JSON-NEXT:        "index": 0
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "FunctionDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 3338,
// JSON-NEXT:         "col": 21,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 3333,
// JSON-NEXT:          "col": 16,
// JSON-NEXT:          "tokLen": 4
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 3343,
// JSON-NEXT:          "col": 26,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "name": "f",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "void ()"
// JSON-NEXT:        },
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "CompoundStmt",
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 3342,
// JSON-NEXT:            "col": 25,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 3343,
// JSON-NEXT:            "col": 26,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          }
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "FunctionDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 3338,
// JSON-NEXT:         "col": 21,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 3333,
// JSON-NEXT:          "col": 16,
// JSON-NEXT:          "tokLen": 4
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 3343,
// JSON-NEXT:          "col": 26,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "isUsed": true,
// JSON-NEXT:        "name": "f",
// JSON-NEXT:        "mangledName": "_ZN5test51fILl0EEEvv",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "void ()"
// JSON-NEXT:        },
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "kind": "TemplateArgument",
// JSON-NEXT:          "value": 0
// JSON-NEXT:         },
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "CompoundStmt",
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 3342,
// JSON-NEXT:            "col": 25,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 3343,
// JSON-NEXT:            "col": 26,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          }
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       }
// JSON-NEXT:      ]
// JSON-NEXT:     },
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "VarDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": 3352,
// JSON-NEXT:       "line": 107,
// JSON-NEXT:       "col": 8,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 3345,
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 4
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 3362,
// JSON-NEXT:        "col": 18,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "name": "p",
// JSON-NEXT:      "mangledName": "_ZN5test51pE",
// JSON-NEXT:      "type": {
// JSON-NEXT:       "qualType": "void (*)()"
// JSON-NEXT:      },
// JSON-NEXT:      "init": "c",
// JSON-NEXT:      "inner": [
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "ImplicitCastExpr",
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 3359,
// JSON-NEXT:          "col": 15,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 3362,
// JSON-NEXT:          "col": 18,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "void (*)()"
// JSON-NEXT:        },
// JSON-NEXT:        "valueCategory": "prvalue",
// JSON-NEXT:        "castKind": "FunctionToPointerDecay",
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "DeclRefExpr",
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 3359,
// JSON-NEXT:            "col": 15,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 3362,
// JSON-NEXT:            "col": 18,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "void ()"
// JSON-NEXT:          },
// JSON-NEXT:          "valueCategory": "lvalue",
// JSON-NEXT:          "referencedDecl": {
// JSON-NEXT:           "id": "0x{{.*}}",
// JSON-NEXT:           "kind": "FunctionDecl",
// JSON-NEXT:           "name": "f",
// JSON-NEXT:           "type": {
// JSON-NEXT:            "qualType": "void ()"
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "foundReferencedDecl": {
// JSON-NEXT:           "id": "0x{{.*}}",
// JSON-NEXT:           "kind": "FunctionTemplateDecl",
// JSON-NEXT:           "name": "f"
// JSON-NEXT:          }
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       }
// JSON-NEXT:      ]
// JSON-NEXT:     },
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "FunctionTemplateDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": 3393,
// JSON-NEXT:       "line": 108,
// JSON-NEXT:       "col": 29,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 3365,
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 3398,
// JSON-NEXT:        "col": 34,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "name": "f",
// JSON-NEXT:      "inner": [
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "NonTypeTemplateParmDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 3383,
// JSON-NEXT:         "col": 19,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 3374,
// JSON-NEXT:          "col": 10,
// JSON-NEXT:          "tokLen": 8
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 3385,
// JSON-NEXT:          "col": 21,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "unsigned int"
// JSON-NEXT:        },
// JSON-NEXT:        "depth": 0,
// JSON-NEXT:        "index": 0,
// JSON-NEXT:        "defaultArg": {
// JSON-NEXT:         "kind": "TemplateArgument",
// JSON-NEXT:         "isExpr": true
// JSON-NEXT:        },
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "kind": "TemplateArgument",
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 3385,
// JSON-NEXT:            "col": 21,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 3385,
// JSON-NEXT:            "col": 21,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "isExpr": true,
// JSON-NEXT:          "inner": [
// JSON-NEXT:           {
// JSON-NEXT:            "id": "0x{{.*}}",
// JSON-NEXT:            "kind": "IntegerLiteral",
// JSON-NEXT:            "range": {
// JSON-NEXT:             "begin": {
// JSON-NEXT:              "offset": 3385,
// JSON-NEXT:              "col": 21,
// JSON-NEXT:              "tokLen": 1
// JSON-NEXT:             },
// JSON-NEXT:             "end": {
// JSON-NEXT:              "offset": 3385,
// JSON-NEXT:              "col": 21,
// JSON-NEXT:              "tokLen": 1
// JSON-NEXT:             }
// JSON-NEXT:            },
// JSON-NEXT:            "type": {
// JSON-NEXT:             "qualType": "int"
// JSON-NEXT:            },
// JSON-NEXT:            "valueCategory": "prvalue",
// JSON-NEXT:            "value": "0"
// JSON-NEXT:           }
// JSON-NEXT:          ]
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "FunctionDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 3393,
// JSON-NEXT:         "col": 29,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 3388,
// JSON-NEXT:          "col": 24,
// JSON-NEXT:          "tokLen": 4
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 3398,
// JSON-NEXT:          "col": 34,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "name": "f",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "void ()"
// JSON-NEXT:        },
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "CompoundStmt",
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 3397,
// JSON-NEXT:            "col": 33,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 3398,
// JSON-NEXT:            "col": 34,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          }
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "FunctionDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 3393,
// JSON-NEXT:         "col": 29,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 3388,
// JSON-NEXT:          "col": 24,
// JSON-NEXT:          "tokLen": 4
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 3398,
// JSON-NEXT:          "col": 34,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "isUsed": true,
// JSON-NEXT:        "name": "f",
// JSON-NEXT:        "mangledName": "_ZN5test51fILj0EEEvv",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "void ()"
// JSON-NEXT:        },
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "kind": "TemplateArgument",
// JSON-NEXT:          "value": 0
// JSON-NEXT:         },
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "CompoundStmt",
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 3397,
// JSON-NEXT:            "col": 33,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 3398,
// JSON-NEXT:            "col": 34,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          }
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       }
// JSON-NEXT:      ]
// JSON-NEXT:     },
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "VarDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": 3407,
// JSON-NEXT:       "line": 109,
// JSON-NEXT:       "col": 8,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 3400,
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 4
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 3416,
// JSON-NEXT:        "col": 17,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "name": "q",
// JSON-NEXT:      "mangledName": "_ZN5test51qE",
// JSON-NEXT:      "type": {
// JSON-NEXT:       "qualType": "void (*)()"
// JSON-NEXT:      },
// JSON-NEXT:      "init": "c",
// JSON-NEXT:      "inner": [
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "ImplicitCastExpr",
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 3414,
// JSON-NEXT:          "col": 15,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 3416,
// JSON-NEXT:          "col": 17,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "void (*)()"
// JSON-NEXT:        },
// JSON-NEXT:        "valueCategory": "prvalue",
// JSON-NEXT:        "castKind": "FunctionToPointerDecay",
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "DeclRefExpr",
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 3414,
// JSON-NEXT:            "col": 15,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 3416,
// JSON-NEXT:            "col": 17,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "void ()"
// JSON-NEXT:          },
// JSON-NEXT:          "valueCategory": "lvalue",
// JSON-NEXT:          "referencedDecl": {
// JSON-NEXT:           "id": "0x{{.*}}",
// JSON-NEXT:           "kind": "FunctionDecl",
// JSON-NEXT:           "name": "f",
// JSON-NEXT:           "type": {
// JSON-NEXT:            "qualType": "void ()"
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "foundReferencedDecl": {
// JSON-NEXT:           "id": "0x{{.*}}",
// JSON-NEXT:           "kind": "FunctionTemplateDecl",
// JSON-NEXT:           "name": "f"
// JSON-NEXT:          }
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       }
// JSON-NEXT:      ]
// JSON-NEXT:     }
// JSON-NEXT:    ]
// JSON-NEXT:   },
// JSON-NEXT:   {
// JSON-NEXT:    "id": "0x{{.*}}",
// JSON-NEXT:    "kind": "NamespaceDecl",
// JSON-NEXT:    "loc": {
// JSON-NEXT:     "offset": 3645,
// JSON-NEXT:     "line": 116,
// JSON-NEXT:     "col": 11,
// JSON-NEXT:     "tokLen": 5
// JSON-NEXT:    },
// JSON-NEXT:    "range": {
// JSON-NEXT:     "begin": {
// JSON-NEXT:      "offset": 3635,
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 9
// JSON-NEXT:     },
// JSON-NEXT:     "end": {
// JSON-NEXT:      "offset": 4000,
// JSON-NEXT:      "line": 128,
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 1
// JSON-NEXT:     }
// JSON-NEXT:    },
// JSON-NEXT:    "name": "test6",
// JSON-NEXT:    "inner": [
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "VarTemplateDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": 3687,
// JSON-NEXT:       "line": 118,
// JSON-NEXT:       "col": 16,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 3653,
// JSON-NEXT:        "line": 117,
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 3691,
// JSON-NEXT:        "line": 118,
// JSON-NEXT:        "col": 20,
// JSON-NEXT:        "tokLen": 4
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "name": "C",
// JSON-NEXT:      "inner": [
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "TemplateTypeParmDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 3669,
// JSON-NEXT:         "line": 117,
// JSON-NEXT:         "col": 17,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 3663,
// JSON-NEXT:          "col": 11,
// JSON-NEXT:          "tokLen": 5
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 3669,
// JSON-NEXT:          "col": 17,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "name": "D",
// JSON-NEXT:        "tagUsed": "class",
// JSON-NEXT:        "depth": 0,
// JSON-NEXT:        "index": 0
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "VarDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 3687,
// JSON-NEXT:         "line": 118,
// JSON-NEXT:         "col": 16,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 3672,
// JSON-NEXT:          "col": 1,
// JSON-NEXT:          "tokLen": 9
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 3691,
// JSON-NEXT:          "col": 20,
// JSON-NEXT:          "tokLen": 4
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "name": "C",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "const bool"
// JSON-NEXT:        },
// JSON-NEXT:        "constexpr": true,
// JSON-NEXT:        "init": "c",
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "CXXBoolLiteralExpr",
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 3691,
// JSON-NEXT:            "col": 20,
// JSON-NEXT:            "tokLen": 4
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 3691,
// JSON-NEXT:            "col": 20,
// JSON-NEXT:            "tokLen": 4
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "bool"
// JSON-NEXT:          },
// JSON-NEXT:          "valueCategory": "prvalue",
// JSON-NEXT:          "value": true
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       }
// JSON-NEXT:      ]
// JSON-NEXT:     },
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "FunctionTemplateDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": 3724,
// JSON-NEXT:       "line": 121,
// JSON-NEXT:       "col": 6,
// JSON-NEXT:       "tokLen": 4
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 3698,
// JSON-NEXT:        "line": 120,
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 3998,
// JSON-NEXT:        "line": 127,
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "name": "func",
// JSON-NEXT:      "inner": [
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "TemplateTypeParmDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 3714,
// JSON-NEXT:         "line": 120,
// JSON-NEXT:         "col": 17,
// JSON-NEXT:         "tokLen": 3
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 3708,
// JSON-NEXT:          "col": 11,
// JSON-NEXT:          "tokLen": 5
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 3714,
// JSON-NEXT:          "col": 17,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "isReferenced": true,
// JSON-NEXT:        "name": "Key",
// JSON-NEXT:        "tagUsed": "class",
// JSON-NEXT:        "depth": 0,
// JSON-NEXT:        "index": 0
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "FunctionDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 3724,
// JSON-NEXT:         "line": 121,
// JSON-NEXT:         "col": 6,
// JSON-NEXT:         "tokLen": 4
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 3719,
// JSON-NEXT:          "col": 1,
// JSON-NEXT:          "tokLen": 4
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 3998,
// JSON-NEXT:          "line": 127,
// JSON-NEXT:          "col": 1,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "name": "func",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "void ()"
// JSON-NEXT:        },
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "CompoundStmt",
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 3731,
// JSON-NEXT:            "line": 121,
// JSON-NEXT:            "col": 13,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 3998,
// JSON-NEXT:            "line": 127,
// JSON-NEXT:            "col": 1,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "inner": [
// JSON-NEXT:           {
// JSON-NEXT:            "id": "0x{{.*}}",
// JSON-NEXT:            "kind": "UnresolvedLookupExpr",
// JSON-NEXT:            "range": {
// JSON-NEXT:             "begin": {
// JSON-NEXT:              "offset": 3735,
// JSON-NEXT:              "line": 122,
// JSON-NEXT:              "col": 3,
// JSON-NEXT:              "tokLen": 1
// JSON-NEXT:             },
// JSON-NEXT:             "end": {
// JSON-NEXT:              "offset": 3740,
// JSON-NEXT:              "col": 8,
// JSON-NEXT:              "tokLen": 1
// JSON-NEXT:             }
// JSON-NEXT:            },
// JSON-NEXT:            "type": {
// JSON-NEXT:             "qualType": "<dependent type>"
// JSON-NEXT:            },
// JSON-NEXT:            "valueCategory": "lvalue",
// JSON-NEXT:            "usesADL": false,
// JSON-NEXT:            "name": "C",
// JSON-NEXT:            "lookups": [
// JSON-NEXT:             {
// JSON-NEXT:              "id": "0x{{.*}}",
// JSON-NEXT:              "kind": "VarTemplateDecl",
// JSON-NEXT:              "name": "C"
// JSON-NEXT:             }
// JSON-NEXT:            ],
// JSON-NEXT:            "inner": [
// JSON-NEXT:             {
// JSON-NEXT:              "kind": "TemplateArgument",
// JSON-NEXT:              "type": {
// JSON-NEXT:               "qualType": "Key"
// JSON-NEXT:              },
// JSON-NEXT:              "inner": [
// JSON-NEXT:               {
// JSON-NEXT:                "id": "0x{{.*}}",
// JSON-NEXT:                "kind": "TemplateTypeParmType",
// JSON-NEXT:                "type": {
// JSON-NEXT:                 "qualType": "Key"
// JSON-NEXT:                },
// JSON-NEXT:                "isDependent": true,
// JSON-NEXT:                "isInstantiationDependent": true,
// JSON-NEXT:                "depth": 0,
// JSON-NEXT:                "index": 0,
// JSON-NEXT:                "decl": {
// JSON-NEXT:                 "id": "0x{{.*}}",
// JSON-NEXT:                 "kind": "TemplateTypeParmDecl",
// JSON-NEXT:                 "name": "Key"
// JSON-NEXT:                }
// JSON-NEXT:               }
// JSON-NEXT:              ]
// JSON-NEXT:             }
// JSON-NEXT:            ]
// JSON-NEXT:           }
// JSON-NEXT:          ]
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       }
// JSON-NEXT:      ]
// JSON-NEXT:     }
// JSON-NEXT:    ]
// JSON-NEXT:   },
// JSON-NEXT:   {
// JSON-NEXT:    "id": "0x{{.*}}",
// JSON-NEXT:    "kind": "NamespaceDecl",
// JSON-NEXT:    "loc": {
// JSON-NEXT:     "offset": 4013,
// JSON-NEXT:     "line": 130,
// JSON-NEXT:     "col": 11,
// JSON-NEXT:     "tokLen": 5
// JSON-NEXT:    },
// JSON-NEXT:    "range": {
// JSON-NEXT:     "begin": {
// JSON-NEXT:      "offset": 4003,
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 9
// JSON-NEXT:     },
// JSON-NEXT:     "end": {
// JSON-NEXT:      "offset": 4308,
// JSON-NEXT:      "line": 136,
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 1
// JSON-NEXT:     }
// JSON-NEXT:    },
// JSON-NEXT:    "name": "test7",
// JSON-NEXT:    "inner": [
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "ClassTemplateDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": 4066,
// JSON-NEXT:       "line": 131,
// JSON-NEXT:       "col": 46,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 4023,
// JSON-NEXT:        "col": 3,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 4069,
// JSON-NEXT:        "col": 49,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "name": "A",
// JSON-NEXT:      "inner": [
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "TemplateTemplateParmDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 4055,
// JSON-NEXT:         "col": 35,
// JSON-NEXT:         "tokLen": 2
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 4033,
// JSON-NEXT:          "col": 13,
// JSON-NEXT:          "tokLen": 8
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 4055,
// JSON-NEXT:          "col": 35,
// JSON-NEXT:          "tokLen": 2
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "name": "TT",
// JSON-NEXT:        "depth": 0,
// JSON-NEXT:        "index": 0,
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "TemplateTypeParmDecl",
// JSON-NEXT:          "loc": {
// JSON-NEXT:           "offset": 4047,
// JSON-NEXT:           "col": 27,
// JSON-NEXT:           "tokLen": 1
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 4042,
// JSON-NEXT:            "col": 22,
// JSON-NEXT:            "tokLen": 5
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 4042,
// JSON-NEXT:            "col": 22,
// JSON-NEXT:            "tokLen": 5
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "tagUsed": "class",
// JSON-NEXT:          "depth": 1,
// JSON-NEXT:          "index": 0
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "CXXRecordDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 4066,
// JSON-NEXT:         "col": 46,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 4059,
// JSON-NEXT:          "col": 39,
// JSON-NEXT:          "tokLen": 6
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 4069,
// JSON-NEXT:          "col": 49,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "name": "A",
// JSON-NEXT:        "tagUsed": "struct",
// JSON-NEXT:        "completeDefinition": true,
// JSON-NEXT:        "definitionData": {
// JSON-NEXT:         "canConstDefaultInit": true,
// JSON-NEXT:         "copyAssign": {
// JSON-NEXT:          "hasConstParam": true,
// JSON-NEXT:          "implicitHasConstParam": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "simple": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         },
// JSON-NEXT:         "copyCtor": {
// JSON-NEXT:          "hasConstParam": true,
// JSON-NEXT:          "implicitHasConstParam": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "simple": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         },
// JSON-NEXT:         "defaultCtor": {
// JSON-NEXT:          "defaultedIsConstexpr": true,
// JSON-NEXT:          "exists": true,
// JSON-NEXT:          "isConstexpr": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         },
// JSON-NEXT:         "dtor": {
// JSON-NEXT:          "irrelevant": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "simple": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         },
// JSON-NEXT:         "hasConstexprNonCopyMoveConstructor": true,
// JSON-NEXT:         "isAggregate": true,
// JSON-NEXT:         "isEmpty": true,
// JSON-NEXT:         "isLiteral": true,
// JSON-NEXT:         "isPOD": true,
// JSON-NEXT:         "isStandardLayout": true,
// JSON-NEXT:         "isTrivial": true,
// JSON-NEXT:         "isTriviallyCopyable": true,
// JSON-NEXT:         "moveAssign": {
// JSON-NEXT:          "exists": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "simple": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         },
// JSON-NEXT:         "moveCtor": {
// JSON-NEXT:          "exists": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "simple": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "CXXRecordDecl",
// JSON-NEXT:          "loc": {
// JSON-NEXT:           "offset": 4066,
// JSON-NEXT:           "col": 46,
// JSON-NEXT:           "tokLen": 1
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 4059,
// JSON-NEXT:            "col": 39,
// JSON-NEXT:            "tokLen": 6
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 4066,
// JSON-NEXT:            "col": 46,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "isImplicit": true,
// JSON-NEXT:          "name": "A",
// JSON-NEXT:          "tagUsed": "struct"
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "ClassTemplateSpecializationDecl",
// JSON-NEXT:        "name": "A"
// JSON-NEXT:       }
// JSON-NEXT:      ]
// JSON-NEXT:     },
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "ClassTemplateDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": 4100,
// JSON-NEXT:       "line": 132,
// JSON-NEXT:       "col": 29,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 4074,
// JSON-NEXT:        "col": 3,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 4103,
// JSON-NEXT:        "col": 32,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "name": "B",
// JSON-NEXT:      "inner": [
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "TemplateTypeParmDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 4092,
// JSON-NEXT:         "col": 21,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 4084,
// JSON-NEXT:          "col": 13,
// JSON-NEXT:          "tokLen": 5
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 4084,
// JSON-NEXT:          "col": 13,
// JSON-NEXT:          "tokLen": 5
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "tagUsed": "class",
// JSON-NEXT:        "depth": 0,
// JSON-NEXT:        "index": 0,
// JSON-NEXT:        "isParameterPack": true
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "CXXRecordDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 4100,
// JSON-NEXT:         "col": 29,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 4094,
// JSON-NEXT:          "col": 23,
// JSON-NEXT:          "tokLen": 5
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 4103,
// JSON-NEXT:          "col": 32,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "name": "B",
// JSON-NEXT:        "tagUsed": "class",
// JSON-NEXT:        "completeDefinition": true,
// JSON-NEXT:        "definitionData": {
// JSON-NEXT:         "canConstDefaultInit": true,
// JSON-NEXT:         "copyAssign": {
// JSON-NEXT:          "hasConstParam": true,
// JSON-NEXT:          "implicitHasConstParam": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "simple": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         },
// JSON-NEXT:         "copyCtor": {
// JSON-NEXT:          "hasConstParam": true,
// JSON-NEXT:          "implicitHasConstParam": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "simple": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         },
// JSON-NEXT:         "defaultCtor": {
// JSON-NEXT:          "defaultedIsConstexpr": true,
// JSON-NEXT:          "exists": true,
// JSON-NEXT:          "isConstexpr": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         },
// JSON-NEXT:         "dtor": {
// JSON-NEXT:          "irrelevant": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "simple": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         },
// JSON-NEXT:         "hasConstexprNonCopyMoveConstructor": true,
// JSON-NEXT:         "isAggregate": true,
// JSON-NEXT:         "isEmpty": true,
// JSON-NEXT:         "isLiteral": true,
// JSON-NEXT:         "isPOD": true,
// JSON-NEXT:         "isStandardLayout": true,
// JSON-NEXT:         "isTrivial": true,
// JSON-NEXT:         "isTriviallyCopyable": true,
// JSON-NEXT:         "moveAssign": {
// JSON-NEXT:          "exists": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "simple": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         },
// JSON-NEXT:         "moveCtor": {
// JSON-NEXT:          "exists": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "simple": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "CXXRecordDecl",
// JSON-NEXT:          "loc": {
// JSON-NEXT:           "offset": 4100,
// JSON-NEXT:           "col": 29,
// JSON-NEXT:           "tokLen": 1
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 4094,
// JSON-NEXT:            "col": 23,
// JSON-NEXT:            "tokLen": 5
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 4100,
// JSON-NEXT:            "col": 29,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "isImplicit": true,
// JSON-NEXT:          "name": "B",
// JSON-NEXT:          "tagUsed": "class"
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       }
// JSON-NEXT:      ]
// JSON-NEXT:     },
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "ClassTemplateSpecializationDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": 4124,
// JSON-NEXT:       "line": 133,
// JSON-NEXT:       "col": 19,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 4108,
// JSON-NEXT:        "col": 3,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 4127,
// JSON-NEXT:        "col": 22,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "name": "A",
// JSON-NEXT:      "tagUsed": "struct",
// JSON-NEXT:      "completeDefinition": true,
// JSON-NEXT:      "strict-pack-match": true,
// JSON-NEXT:      "definitionData": {
// JSON-NEXT:       "canConstDefaultInit": true,
// JSON-NEXT:       "canPassInRegisters": true,
// JSON-NEXT:       "copyAssign": {
// JSON-NEXT:        "hasConstParam": true,
// JSON-NEXT:        "implicitHasConstParam": true,
// JSON-NEXT:        "needsImplicit": true,
// JSON-NEXT:        "simple": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       },
// JSON-NEXT:       "copyCtor": {
// JSON-NEXT:        "hasConstParam": true,
// JSON-NEXT:        "implicitHasConstParam": true,
// JSON-NEXT:        "needsImplicit": true,
// JSON-NEXT:        "simple": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       },
// JSON-NEXT:       "defaultCtor": {
// JSON-NEXT:        "defaultedIsConstexpr": true,
// JSON-NEXT:        "exists": true,
// JSON-NEXT:        "isConstexpr": true,
// JSON-NEXT:        "needsImplicit": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       },
// JSON-NEXT:       "dtor": {
// JSON-NEXT:        "irrelevant": true,
// JSON-NEXT:        "needsImplicit": true,
// JSON-NEXT:        "simple": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       },
// JSON-NEXT:       "hasConstexprNonCopyMoveConstructor": true,
// JSON-NEXT:       "isAggregate": true,
// JSON-NEXT:       "isEmpty": true,
// JSON-NEXT:       "isLiteral": true,
// JSON-NEXT:       "isPOD": true,
// JSON-NEXT:       "isStandardLayout": true,
// JSON-NEXT:       "isTrivial": true,
// JSON-NEXT:       "isTriviallyCopyable": true,
// JSON-NEXT:       "moveAssign": {
// JSON-NEXT:        "exists": true,
// JSON-NEXT:        "needsImplicit": true,
// JSON-NEXT:        "simple": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       },
// JSON-NEXT:       "moveCtor": {
// JSON-NEXT:        "exists": true,
// JSON-NEXT:        "needsImplicit": true,
// JSON-NEXT:        "simple": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "inner": [
// JSON-NEXT:       {
// JSON-NEXT:        "kind": "TemplateArgument"
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "CXXRecordDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 4066,
// JSON-NEXT:         "line": 131,
// JSON-NEXT:         "col": 46,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 4059,
// JSON-NEXT:          "col": 39,
// JSON-NEXT:          "tokLen": 6
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 4066,
// JSON-NEXT:          "col": 46,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "isImplicit": true,
// JSON-NEXT:        "name": "A",
// JSON-NEXT:        "tagUsed": "struct"
// JSON-NEXT:       }
// JSON-NEXT:      ]
// JSON-NEXT:     }
// JSON-NEXT:    ]
// JSON-NEXT:   },
// JSON-NEXT:   {
// JSON-NEXT:    "id": "0x{{.*}}",
// JSON-NEXT:    "kind": "NamespaceDecl",
// JSON-NEXT:    "loc": {
// JSON-NEXT:     "offset": 4339,
// JSON-NEXT:     "line": 138,
// JSON-NEXT:     "col": 11,
// JSON-NEXT:     "tokLen": 5
// JSON-NEXT:    },
// JSON-NEXT:    "range": {
// JSON-NEXT:     "begin": {
// JSON-NEXT:      "offset": 4329,
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 9
// JSON-NEXT:     },
// JSON-NEXT:     "end": {
// JSON-NEXT:      "offset": 4648,
// JSON-NEXT:      "line": 147,
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 1
// JSON-NEXT:     }
// JSON-NEXT:    },
// JSON-NEXT:    "name": "test8",
// JSON-NEXT:    "inner": [
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "ClassTemplateDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": 4379,
// JSON-NEXT:       "line": 140,
// JSON-NEXT:       "col": 8,
// JSON-NEXT:       "tokLen": 8
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 4347,
// JSON-NEXT:        "line": 139,
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 4379,
// JSON-NEXT:        "line": 140,
// JSON-NEXT:        "col": 8,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "name": "pr126341",
// JSON-NEXT:      "inner": [
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "NonTypeTemplateParmDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 4369,
// JSON-NEXT:         "line": 139,
// JSON-NEXT:         "col": 23,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 4356,
// JSON-NEXT:          "col": 10,
// JSON-NEXT:          "tokLen": 8
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 4369,
// JSON-NEXT:          "col": 23,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "name": "x",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "_Complex int"
// JSON-NEXT:        },
// JSON-NEXT:        "depth": 0,
// JSON-NEXT:        "index": 0
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "CXXRecordDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 4379,
// JSON-NEXT:         "line": 140,
// JSON-NEXT:         "col": 8,
// JSON-NEXT:         "tokLen": 8
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 4372,
// JSON-NEXT:          "col": 1,
// JSON-NEXT:          "tokLen": 6
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 4379,
// JSON-NEXT:          "col": 8,
// JSON-NEXT:          "tokLen": 8
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "name": "pr126341",
// JSON-NEXT:        "tagUsed": "struct"
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "ClassTemplateSpecializationDecl",
// JSON-NEXT:        "name": "pr126341"
// JSON-NEXT:       }
// JSON-NEXT:      ]
// JSON-NEXT:     },
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "ClassTemplateSpecializationDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": 4407,
// JSON-NEXT:       "line": 142,
// JSON-NEXT:       "col": 8,
// JSON-NEXT:       "tokLen": 8
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 4389,
// JSON-NEXT:        "line": 141,
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 4422,
// JSON-NEXT:        "line": 142,
// JSON-NEXT:        "col": 23,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "name": "pr126341",
// JSON-NEXT:      "tagUsed": "struct",
// JSON-NEXT:      "inner": [
// JSON-NEXT:       {
// JSON-NEXT:        "kind": "TemplateArgument",
// JSON-NEXT:        "value": "1+2i"
// JSON-NEXT:       }
// JSON-NEXT:      ]
// JSON-NEXT:     }
// JSON-NEXT:    ]
// JSON-NEXT:   },
// JSON-NEXT:   {
// JSON-NEXT:    "id": "0x{{.*}}",
// JSON-NEXT:    "kind": "NamespaceDecl",
// JSON-NEXT:    "loc": {
// JSON-NEXT:     "offset": 4680,
// JSON-NEXT:     "line": 149,
// JSON-NEXT:     "col": 11,
// JSON-NEXT:     "tokLen": 28
// JSON-NEXT:    },
// JSON-NEXT:    "range": {
// JSON-NEXT:     "begin": {
// JSON-NEXT:      "offset": 4670,
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 9
// JSON-NEXT:     },
// JSON-NEXT:     "end": {
// JSON-NEXT:      "offset": 5303,
// JSON-NEXT:      "line": 158,
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 1
// JSON-NEXT:     }
// JSON-NEXT:    },
// JSON-NEXT:    "name": "TestMemberPointerPartialSpec",
// JSON-NEXT:    "inner": [
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "ClassTemplateDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": 4737,
// JSON-NEXT:       "line": 150,
// JSON-NEXT:       "col": 27,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 4713,
// JSON-NEXT:        "col": 3,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 4737,
// JSON-NEXT:        "col": 27,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "name": "A",
// JSON-NEXT:      "inner": [
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "TemplateTypeParmDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 4728,
// JSON-NEXT:         "col": 18,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 4723,
// JSON-NEXT:          "col": 13,
// JSON-NEXT:          "tokLen": 5
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 4723,
// JSON-NEXT:          "col": 13,
// JSON-NEXT:          "tokLen": 5
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "tagUsed": "class",
// JSON-NEXT:        "depth": 0,
// JSON-NEXT:        "index": 0
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "CXXRecordDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 4737,
// JSON-NEXT:         "col": 27,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 4730,
// JSON-NEXT:          "col": 20,
// JSON-NEXT:          "tokLen": 6
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 4737,
// JSON-NEXT:          "col": 27,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "name": "A",
// JSON-NEXT:        "tagUsed": "struct"
// JSON-NEXT:       }
// JSON-NEXT:      ]
// JSON-NEXT:     },
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "ClassTemplatePartialSpecializationDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": 4779,
// JSON-NEXT:       "line": 151,
// JSON-NEXT:       "col": 40,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 4742,
// JSON-NEXT:        "col": 3,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 4789,
// JSON-NEXT:        "col": 50,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "name": "A",
// JSON-NEXT:      "tagUsed": "struct",
// JSON-NEXT:      "inner": [
// JSON-NEXT:       {
// JSON-NEXT:        "kind": "TemplateArgument",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "type-parameter-0-0 type-parameter-0-1::*"
// JSON-NEXT:        },
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "MemberPointerType",
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "type-parameter-0-0 type-parameter-0-1::*"
// JSON-NEXT:          },
// JSON-NEXT:          "isDependent": true,
// JSON-NEXT:          "isInstantiationDependent": true,
// JSON-NEXT:          "isData": true,
// JSON-NEXT:          "inner": [
// JSON-NEXT:           {
// JSON-NEXT:            "id": "0x{{.*}}",
// JSON-NEXT:            "kind": "TemplateTypeParmType",
// JSON-NEXT:            "type": {
// JSON-NEXT:             "qualType": "type-parameter-0-1"
// JSON-NEXT:            },
// JSON-NEXT:            "isDependent": true,
// JSON-NEXT:            "isInstantiationDependent": true,
// JSON-NEXT:            "depth": 0,
// JSON-NEXT:            "index": 1,
// JSON-NEXT:            "decl": {
// JSON-NEXT:             "id": "0x0"
// JSON-NEXT:            }
// JSON-NEXT:           },
// JSON-NEXT:           {
// JSON-NEXT:            "id": "0x{{.*}}",
// JSON-NEXT:            "kind": "TemplateTypeParmType",
// JSON-NEXT:            "type": {
// JSON-NEXT:             "qualType": "type-parameter-0-0"
// JSON-NEXT:            },
// JSON-NEXT:            "isDependent": true,
// JSON-NEXT:            "isInstantiationDependent": true,
// JSON-NEXT:            "depth": 0,
// JSON-NEXT:            "index": 0,
// JSON-NEXT:            "decl": {
// JSON-NEXT:             "id": "0x0"
// JSON-NEXT:            }
// JSON-NEXT:           }
// JSON-NEXT:          ]
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "TemplateTypeParmDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 4758,
// JSON-NEXT:         "col": 19,
// JSON-NEXT:         "tokLen": 2
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 4752,
// JSON-NEXT:          "col": 13,
// JSON-NEXT:          "tokLen": 5
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 4758,
// JSON-NEXT:          "col": 19,
// JSON-NEXT:          "tokLen": 2
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "isReferenced": true,
// JSON-NEXT:        "name": "T1",
// JSON-NEXT:        "tagUsed": "class",
// JSON-NEXT:        "depth": 0,
// JSON-NEXT:        "index": 0
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "TemplateTypeParmDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 4768,
// JSON-NEXT:         "col": 29,
// JSON-NEXT:         "tokLen": 2
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 4762,
// JSON-NEXT:          "col": 23,
// JSON-NEXT:          "tokLen": 5
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 4768,
// JSON-NEXT:          "col": 29,
// JSON-NEXT:          "tokLen": 2
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "name": "T2",
// JSON-NEXT:        "tagUsed": "class",
// JSON-NEXT:        "depth": 0,
// JSON-NEXT:        "index": 1
// JSON-NEXT:       }
// JSON-NEXT:      ]
// JSON-NEXT:     }
// JSON-NEXT:    ]
// JSON-NEXT:   },
// JSON-NEXT:   {
// JSON-NEXT:    "id": "0x{{.*}}",
// JSON-NEXT:    "kind": "NamespaceDecl",
// JSON-NEXT:    "loc": {
// JSON-NEXT:     "offset": 5358,
// JSON-NEXT:     "line": 160,
// JSON-NEXT:     "col": 11,
// JSON-NEXT:     "tokLen": 26
// JSON-NEXT:    },
// JSON-NEXT:    "range": {
// JSON-NEXT:     "begin": {
// JSON-NEXT:      "offset": 5348,
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 9
// JSON-NEXT:     },
// JSON-NEXT:     "end": {
// JSON-NEXT:      "offset": 6425,
// JSON-NEXT:      "line": 180,
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 1
// JSON-NEXT:     }
// JSON-NEXT:    },
// JSON-NEXT:    "name": "TestDependentMemberPointer",
// JSON-NEXT:    "inner": [
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "ClassTemplateDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": 5415,
// JSON-NEXT:       "line": 161,
// JSON-NEXT:       "col": 29,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 5389,
// JSON-NEXT:        "col": 3,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 5516,
// JSON-NEXT:        "line": 165,
// JSON-NEXT:        "col": 3,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "name": "A",
// JSON-NEXT:      "inner": [
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "TemplateTypeParmDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 5405,
// JSON-NEXT:         "line": 161,
// JSON-NEXT:         "col": 19,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 5399,
// JSON-NEXT:          "col": 13,
// JSON-NEXT:          "tokLen": 5
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 5405,
// JSON-NEXT:          "col": 19,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "name": "U",
// JSON-NEXT:        "tagUsed": "class",
// JSON-NEXT:        "depth": 0,
// JSON-NEXT:        "index": 0
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "CXXRecordDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 5415,
// JSON-NEXT:         "col": 29,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 5408,
// JSON-NEXT:          "col": 22,
// JSON-NEXT:          "tokLen": 6
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 5516,
// JSON-NEXT:          "line": 165,
// JSON-NEXT:          "col": 3,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "name": "A",
// JSON-NEXT:        "tagUsed": "struct",
// JSON-NEXT:        "completeDefinition": true,
// JSON-NEXT:        "definitionData": {
// JSON-NEXT:         "canConstDefaultInit": true,
// JSON-NEXT:         "copyAssign": {
// JSON-NEXT:          "hasConstParam": true,
// JSON-NEXT:          "implicitHasConstParam": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "simple": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         },
// JSON-NEXT:         "copyCtor": {
// JSON-NEXT:          "hasConstParam": true,
// JSON-NEXT:          "implicitHasConstParam": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "simple": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         },
// JSON-NEXT:         "defaultCtor": {
// JSON-NEXT:          "defaultedIsConstexpr": true,
// JSON-NEXT:          "exists": true,
// JSON-NEXT:          "isConstexpr": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         },
// JSON-NEXT:         "dtor": {
// JSON-NEXT:          "irrelevant": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "simple": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         },
// JSON-NEXT:         "hasConstexprNonCopyMoveConstructor": true,
// JSON-NEXT:         "isAggregate": true,
// JSON-NEXT:         "isEmpty": true,
// JSON-NEXT:         "isLiteral": true,
// JSON-NEXT:         "isPOD": true,
// JSON-NEXT:         "isStandardLayout": true,
// JSON-NEXT:         "isTrivial": true,
// JSON-NEXT:         "isTriviallyCopyable": true,
// JSON-NEXT:         "moveAssign": {
// JSON-NEXT:          "exists": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "simple": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         },
// JSON-NEXT:         "moveCtor": {
// JSON-NEXT:          "exists": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "simple": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "CXXRecordDecl",
// JSON-NEXT:          "loc": {
// JSON-NEXT:           "offset": 5415,
// JSON-NEXT:           "line": 161,
// JSON-NEXT:           "col": 29,
// JSON-NEXT:           "tokLen": 1
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 5408,
// JSON-NEXT:            "col": 22,
// JSON-NEXT:            "tokLen": 6
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 5415,
// JSON-NEXT:            "col": 29,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "isImplicit": true,
// JSON-NEXT:          "name": "A",
// JSON-NEXT:          "tagUsed": "struct"
// JSON-NEXT:         },
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "TypeAliasDecl",
// JSON-NEXT:          "loc": {
// JSON-NEXT:           "offset": 5429,
// JSON-NEXT:           "line": 162,
// JSON-NEXT:           "col": 11,
// JSON-NEXT:           "tokLen": 1
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 5423,
// JSON-NEXT:            "col": 5,
// JSON-NEXT:            "tokLen": 5
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 5440,
// JSON-NEXT:            "col": 22,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "name": "X",
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "int U::*"
// JSON-NEXT:          },
// JSON-NEXT:          "inner": [
// JSON-NEXT:           {
// JSON-NEXT:            "id": "0x{{.*}}",
// JSON-NEXT:            "kind": "MemberPointerType",
// JSON-NEXT:            "type": {
// JSON-NEXT:             "qualType": "int U::*"
// JSON-NEXT:            },
// JSON-NEXT:            "isDependent": true,
// JSON-NEXT:            "isInstantiationDependent": true,
// JSON-NEXT:            "isData": true,
// JSON-NEXT:            "inner": [
// JSON-NEXT:             {
// JSON-NEXT:              "id": "0x{{.*}}",
// JSON-NEXT:              "kind": "TemplateTypeParmType",
// JSON-NEXT:              "type": {
// JSON-NEXT:               "qualType": "U"
// JSON-NEXT:              },
// JSON-NEXT:              "isDependent": true,
// JSON-NEXT:              "isInstantiationDependent": true,
// JSON-NEXT:              "depth": 0,
// JSON-NEXT:              "index": 0,
// JSON-NEXT:              "decl": {
// JSON-NEXT:               "id": "0x{{.*}}",
// JSON-NEXT:               "kind": "TemplateTypeParmDecl",
// JSON-NEXT:               "name": "U"
// JSON-NEXT:              }
// JSON-NEXT:             },
// JSON-NEXT:             {
// JSON-NEXT:              "id": "0x{{.*}}",
// JSON-NEXT:              "kind": "BuiltinType",
// JSON-NEXT:              "type": {
// JSON-NEXT:               "qualType": "int"
// JSON-NEXT:              }
// JSON-NEXT:             }
// JSON-NEXT:            ]
// JSON-NEXT:           }
// JSON-NEXT:          ]
// JSON-NEXT:         },
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "TypeAliasDecl",
// JSON-NEXT:          "loc": {
// JSON-NEXT:           "offset": 5453,
// JSON-NEXT:           "line": 163,
// JSON-NEXT:           "col": 11,
// JSON-NEXT:           "tokLen": 1
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 5447,
// JSON-NEXT:            "col": 5,
// JSON-NEXT:            "tokLen": 5
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 5470,
// JSON-NEXT:            "col": 28,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "name": "Y",
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "int U::test::*"
// JSON-NEXT:          },
// JSON-NEXT:          "inner": [
// JSON-NEXT:           {
// JSON-NEXT:            "id": "0x{{.*}}",
// JSON-NEXT:            "kind": "MemberPointerType",
// JSON-NEXT:            "type": {
// JSON-NEXT:             "qualType": "int U::test::*"
// JSON-NEXT:            },
// JSON-NEXT:            "isDependent": true,
// JSON-NEXT:            "isInstantiationDependent": true,
// JSON-NEXT:            "isData": true,
// JSON-NEXT:            "inner": [
// JSON-NEXT:             {
// JSON-NEXT:              "id": "0x{{.*}}",
// JSON-NEXT:              "kind": "DependentNameType",
// JSON-NEXT:              "type": {
// JSON-NEXT:               "qualType": "U::test"
// JSON-NEXT:              },
// JSON-NEXT:              "isDependent": true,
// JSON-NEXT:              "isInstantiationDependent": true
// JSON-NEXT:             },
// JSON-NEXT:             {
// JSON-NEXT:              "id": "0x{{.*}}",
// JSON-NEXT:              "kind": "BuiltinType",
// JSON-NEXT:              "type": {
// JSON-NEXT:               "qualType": "int"
// JSON-NEXT:              }
// JSON-NEXT:             }
// JSON-NEXT:            ]
// JSON-NEXT:           }
// JSON-NEXT:          ]
// JSON-NEXT:         },
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "TypeAliasDecl",
// JSON-NEXT:          "loc": {
// JSON-NEXT:           "offset": 5483,
// JSON-NEXT:           "line": 164,
// JSON-NEXT:           "col": 11,
// JSON-NEXT:           "tokLen": 1
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 5477,
// JSON-NEXT:            "col": 5,
// JSON-NEXT:            "tokLen": 5
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 5511,
// JSON-NEXT:            "col": 39,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "name": "Z",
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "int U::template V<int>::*"
// JSON-NEXT:          },
// JSON-NEXT:          "inner": [
// JSON-NEXT:           {
// JSON-NEXT:            "id": "0x{{.*}}",
// JSON-NEXT:            "kind": "MemberPointerType",
// JSON-NEXT:            "type": {
// JSON-NEXT:             "qualType": "int U::template V<int>::*"
// JSON-NEXT:            },
// JSON-NEXT:            "isDependent": true,
// JSON-NEXT:            "isInstantiationDependent": true,
// JSON-NEXT:            "isData": true,
// JSON-NEXT:            "inner": [
// JSON-NEXT:             {
// JSON-NEXT:              "id": "0x{{.*}}",
// JSON-NEXT:              "kind": "DependentTemplateSpecializationType",
// JSON-NEXT:              "type": {
// JSON-NEXT:               "qualType": "U::template V<int>"
// JSON-NEXT:              },
// JSON-NEXT:              "isDependent": true,
// JSON-NEXT:              "isInstantiationDependent": true
// JSON-NEXT:             },
// JSON-NEXT:             {
// JSON-NEXT:              "id": "0x{{.*}}",
// JSON-NEXT:              "kind": "BuiltinType",
// JSON-NEXT:              "type": {
// JSON-NEXT:               "qualType": "int"
// JSON-NEXT:              }
// JSON-NEXT:             }
// JSON-NEXT:            ]
// JSON-NEXT:           }
// JSON-NEXT:          ]
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       }
// JSON-NEXT:      ]
// JSON-NEXT:     }
// JSON-NEXT:    ]
// JSON-NEXT:   },
// JSON-NEXT:   {
// JSON-NEXT:    "id": "0x{{.*}}",
// JSON-NEXT:    "kind": "NamespaceDecl",
// JSON-NEXT:    "loc": {
// JSON-NEXT:     "offset": 6478,
// JSON-NEXT:     "line": 182,
// JSON-NEXT:     "col": 11,
// JSON-NEXT:     "tokLen": 19
// JSON-NEXT:    },
// JSON-NEXT:    "range": {
// JSON-NEXT:     "begin": {
// JSON-NEXT:      "offset": 6468,
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 9
// JSON-NEXT:     },
// JSON-NEXT:     "end": {
// JSON-NEXT:      "offset": 9336,
// JSON-NEXT:      "line": 222,
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 1
// JSON-NEXT:     }
// JSON-NEXT:    },
// JSON-NEXT:    "name": "TestPartialSpecNTTP",
// JSON-NEXT:    "inner": [
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "ClassTemplateDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": 6601,
// JSON-NEXT:       "line": 184,
// JSON-NEXT:       "col": 41,
// JSON-NEXT:       "tokLen": 9
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 6563,
// JSON-NEXT:        "col": 3,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 6612,
// JSON-NEXT:        "col": 52,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "name": "Template1",
// JSON-NEXT:      "inner": [
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "TemplateTypeParmDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 6579,
// JSON-NEXT:         "col": 19,
// JSON-NEXT:         "tokLen": 3
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 6573,
// JSON-NEXT:          "col": 13,
// JSON-NEXT:          "tokLen": 5
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 6579,
// JSON-NEXT:          "col": 19,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "name": "TA1",
// JSON-NEXT:        "tagUsed": "class",
// JSON-NEXT:        "depth": 0,
// JSON-NEXT:        "index": 0
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "NonTypeTemplateParmDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 6589,
// JSON-NEXT:         "col": 29,
// JSON-NEXT:         "tokLen": 3
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 6584,
// JSON-NEXT:          "col": 24,
// JSON-NEXT:          "tokLen": 4
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 6589,
// JSON-NEXT:          "col": 29,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "name": "TA2",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "bool"
// JSON-NEXT:        },
// JSON-NEXT:        "depth": 0,
// JSON-NEXT:        "index": 1
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "CXXRecordDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 6601,
// JSON-NEXT:         "col": 41,
// JSON-NEXT:         "tokLen": 9
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 6594,
// JSON-NEXT:          "col": 34,
// JSON-NEXT:          "tokLen": 6
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 6612,
// JSON-NEXT:          "col": 52,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "name": "Template1",
// JSON-NEXT:        "tagUsed": "struct",
// JSON-NEXT:        "completeDefinition": true,
// JSON-NEXT:        "definitionData": {
// JSON-NEXT:         "canConstDefaultInit": true,
// JSON-NEXT:         "copyAssign": {
// JSON-NEXT:          "hasConstParam": true,
// JSON-NEXT:          "implicitHasConstParam": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "simple": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         },
// JSON-NEXT:         "copyCtor": {
// JSON-NEXT:          "hasConstParam": true,
// JSON-NEXT:          "implicitHasConstParam": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "simple": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         },
// JSON-NEXT:         "defaultCtor": {
// JSON-NEXT:          "defaultedIsConstexpr": true,
// JSON-NEXT:          "exists": true,
// JSON-NEXT:          "isConstexpr": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         },
// JSON-NEXT:         "dtor": {
// JSON-NEXT:          "irrelevant": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "simple": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         },
// JSON-NEXT:         "hasConstexprNonCopyMoveConstructor": true,
// JSON-NEXT:         "isAggregate": true,
// JSON-NEXT:         "isEmpty": true,
// JSON-NEXT:         "isLiteral": true,
// JSON-NEXT:         "isPOD": true,
// JSON-NEXT:         "isStandardLayout": true,
// JSON-NEXT:         "isTrivial": true,
// JSON-NEXT:         "isTriviallyCopyable": true,
// JSON-NEXT:         "moveAssign": {
// JSON-NEXT:          "exists": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "simple": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         },
// JSON-NEXT:         "moveCtor": {
// JSON-NEXT:          "exists": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "simple": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "CXXRecordDecl",
// JSON-NEXT:          "loc": {
// JSON-NEXT:           "offset": 6601,
// JSON-NEXT:           "col": 41,
// JSON-NEXT:           "tokLen": 9
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 6594,
// JSON-NEXT:            "col": 34,
// JSON-NEXT:            "tokLen": 6
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 6601,
// JSON-NEXT:            "col": 41,
// JSON-NEXT:            "tokLen": 9
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "isImplicit": true,
// JSON-NEXT:          "name": "Template1",
// JSON-NEXT:          "tagUsed": "struct"
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       }
// JSON-NEXT:      ]
// JSON-NEXT:     },
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "ClassTemplateDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": 6655,
// JSON-NEXT:       "line": 185,
// JSON-NEXT:       "col": 41,
// JSON-NEXT:       "tokLen": 9
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 6617,
// JSON-NEXT:        "col": 3,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 6666,
// JSON-NEXT:        "col": 52,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "name": "Template2",
// JSON-NEXT:      "inner": [
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "TemplateTypeParmDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 6633,
// JSON-NEXT:         "col": 19,
// JSON-NEXT:         "tokLen": 3
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 6627,
// JSON-NEXT:          "col": 13,
// JSON-NEXT:          "tokLen": 5
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 6633,
// JSON-NEXT:          "col": 19,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "name": "TB1",
// JSON-NEXT:        "tagUsed": "class",
// JSON-NEXT:        "depth": 0,
// JSON-NEXT:        "index": 0
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "NonTypeTemplateParmDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 6643,
// JSON-NEXT:         "col": 29,
// JSON-NEXT:         "tokLen": 3
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 6638,
// JSON-NEXT:          "col": 24,
// JSON-NEXT:          "tokLen": 4
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 6643,
// JSON-NEXT:          "col": 29,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "name": "TB2",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "bool"
// JSON-NEXT:        },
// JSON-NEXT:        "depth": 0,
// JSON-NEXT:        "index": 1
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "CXXRecordDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 6655,
// JSON-NEXT:         "col": 41,
// JSON-NEXT:         "tokLen": 9
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 6648,
// JSON-NEXT:          "col": 34,
// JSON-NEXT:          "tokLen": 6
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 6666,
// JSON-NEXT:          "col": 52,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "name": "Template2",
// JSON-NEXT:        "tagUsed": "struct",
// JSON-NEXT:        "completeDefinition": true,
// JSON-NEXT:        "definitionData": {
// JSON-NEXT:         "canConstDefaultInit": true,
// JSON-NEXT:         "copyAssign": {
// JSON-NEXT:          "hasConstParam": true,
// JSON-NEXT:          "implicitHasConstParam": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "simple": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         },
// JSON-NEXT:         "copyCtor": {
// JSON-NEXT:          "hasConstParam": true,
// JSON-NEXT:          "implicitHasConstParam": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "simple": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         },
// JSON-NEXT:         "defaultCtor": {
// JSON-NEXT:          "defaultedIsConstexpr": true,
// JSON-NEXT:          "exists": true,
// JSON-NEXT:          "isConstexpr": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         },
// JSON-NEXT:         "dtor": {
// JSON-NEXT:          "irrelevant": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "simple": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         },
// JSON-NEXT:         "hasConstexprNonCopyMoveConstructor": true,
// JSON-NEXT:         "isAggregate": true,
// JSON-NEXT:         "isEmpty": true,
// JSON-NEXT:         "isLiteral": true,
// JSON-NEXT:         "isPOD": true,
// JSON-NEXT:         "isStandardLayout": true,
// JSON-NEXT:         "isTrivial": true,
// JSON-NEXT:         "isTriviallyCopyable": true,
// JSON-NEXT:         "moveAssign": {
// JSON-NEXT:          "exists": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "simple": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         },
// JSON-NEXT:         "moveCtor": {
// JSON-NEXT:          "exists": true,
// JSON-NEXT:          "needsImplicit": true,
// JSON-NEXT:          "simple": true,
// JSON-NEXT:          "trivial": true
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "CXXRecordDecl",
// JSON-NEXT:          "loc": {
// JSON-NEXT:           "offset": 6655,
// JSON-NEXT:           "col": 41,
// JSON-NEXT:           "tokLen": 9
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 6648,
// JSON-NEXT:            "col": 34,
// JSON-NEXT:            "tokLen": 6
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 6655,
// JSON-NEXT:            "col": 41,
// JSON-NEXT:            "tokLen": 9
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "isImplicit": true,
// JSON-NEXT:          "name": "Template2",
// JSON-NEXT:          "tagUsed": "struct"
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       }
// JSON-NEXT:      ]
// JSON-NEXT:     },
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "ClassTemplatePartialSpecializationDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": 6719,
// JSON-NEXT:       "line": 188,
// JSON-NEXT:       "col": 10,
// JSON-NEXT:       "tokLen": 9
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 6672,
// JSON-NEXT:        "line": 187,
// JSON-NEXT:        "col": 3,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 6753,
// JSON-NEXT:        "line": 188,
// JSON-NEXT:        "col": 44,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "name": "Template2",
// JSON-NEXT:      "tagUsed": "struct",
// JSON-NEXT:      "completeDefinition": true,
// JSON-NEXT:      "definitionData": {
// JSON-NEXT:       "canConstDefaultInit": true,
// JSON-NEXT:       "copyAssign": {
// JSON-NEXT:        "hasConstParam": true,
// JSON-NEXT:        "implicitHasConstParam": true,
// JSON-NEXT:        "needsImplicit": true,
// JSON-NEXT:        "simple": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       },
// JSON-NEXT:       "copyCtor": {
// JSON-NEXT:        "hasConstParam": true,
// JSON-NEXT:        "implicitHasConstParam": true,
// JSON-NEXT:        "needsImplicit": true,
// JSON-NEXT:        "simple": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       },
// JSON-NEXT:       "defaultCtor": {
// JSON-NEXT:        "defaultedIsConstexpr": true,
// JSON-NEXT:        "exists": true,
// JSON-NEXT:        "isConstexpr": true,
// JSON-NEXT:        "needsImplicit": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       },
// JSON-NEXT:       "dtor": {
// JSON-NEXT:        "irrelevant": true,
// JSON-NEXT:        "needsImplicit": true,
// JSON-NEXT:        "simple": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       },
// JSON-NEXT:       "hasConstexprNonCopyMoveConstructor": true,
// JSON-NEXT:       "isAggregate": true,
// JSON-NEXT:       "isEmpty": true,
// JSON-NEXT:       "isLiteral": true,
// JSON-NEXT:       "isPOD": true,
// JSON-NEXT:       "isStandardLayout": true,
// JSON-NEXT:       "isTrivial": true,
// JSON-NEXT:       "isTriviallyCopyable": true,
// JSON-NEXT:       "moveAssign": {
// JSON-NEXT:        "exists": true,
// JSON-NEXT:        "needsImplicit": true,
// JSON-NEXT:        "simple": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       },
// JSON-NEXT:       "moveCtor": {
// JSON-NEXT:        "exists": true,
// JSON-NEXT:        "needsImplicit": true,
// JSON-NEXT:        "simple": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "inner": [
// JSON-NEXT:       {
// JSON-NEXT:        "kind": "TemplateArgument",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "TestPartialSpecNTTP::Template1<type-parameter-0-0, value-parameter-0-1>"
// JSON-NEXT:        },
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "TemplateSpecializationType",
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "TestPartialSpecNTTP::Template1<type-parameter-0-0, value-parameter-0-1>"
// JSON-NEXT:          },
// JSON-NEXT:          "isDependent": true,
// JSON-NEXT:          "isInstantiationDependent": true,
// JSON-NEXT:          "templateName": "TestPartialSpecNTTP::Template1",
// JSON-NEXT:          "inner": [
// JSON-NEXT:           {
// JSON-NEXT:            "kind": "TemplateArgument",
// JSON-NEXT:            "type": {
// JSON-NEXT:             "qualType": "type-parameter-0-0"
// JSON-NEXT:            },
// JSON-NEXT:            "inner": [
// JSON-NEXT:             {
// JSON-NEXT:              "id": "0x{{.*}}",
// JSON-NEXT:              "kind": "TemplateTypeParmType",
// JSON-NEXT:              "type": {
// JSON-NEXT:               "qualType": "type-parameter-0-0"
// JSON-NEXT:              },
// JSON-NEXT:              "isDependent": true,
// JSON-NEXT:              "isInstantiationDependent": true,
// JSON-NEXT:              "depth": 0,
// JSON-NEXT:              "index": 0,
// JSON-NEXT:              "decl": {
// JSON-NEXT:               "id": "0x0"
// JSON-NEXT:              }
// JSON-NEXT:             }
// JSON-NEXT:            ]
// JSON-NEXT:           },
// JSON-NEXT:           {
// JSON-NEXT:            "kind": "TemplateArgument",
// JSON-NEXT:            "isExpr": true,
// JSON-NEXT:            "isCanonical": true,
// JSON-NEXT:            "inner": [
// JSON-NEXT:             {
// JSON-NEXT:              "id": "0x{{.*}}",
// JSON-NEXT:              "kind": "DeclRefExpr",
// JSON-NEXT:              "range": {
// JSON-NEXT:               "begin": {
// JSON-NEXT:                "offset": 6743,
// JSON-NEXT:                "col": 34,
// JSON-NEXT:                "tokLen": 2
// JSON-NEXT:               },
// JSON-NEXT:               "end": {
// JSON-NEXT:                "offset": 6743,
// JSON-NEXT:                "col": 34,
// JSON-NEXT:                "tokLen": 2
// JSON-NEXT:               }
// JSON-NEXT:              },
// JSON-NEXT:              "type": {
// JSON-NEXT:               "qualType": "bool"
// JSON-NEXT:              },
// JSON-NEXT:              "valueCategory": "prvalue",
// JSON-NEXT:              "referencedDecl": {
// JSON-NEXT:               "id": "0x{{.*}}",
// JSON-NEXT:               "kind": "NonTypeTemplateParmDecl",
// JSON-NEXT:               "name": "U2",
// JSON-NEXT:               "type": {
// JSON-NEXT:                "qualType": "bool"
// JSON-NEXT:               }
// JSON-NEXT:              }
// JSON-NEXT:             }
// JSON-NEXT:            ]
// JSON-NEXT:           }
// JSON-NEXT:          ]
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "kind": "TemplateArgument",
// JSON-NEXT:        "isExpr": true,
// JSON-NEXT:        "isCanonical": true,
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "DeclRefExpr",
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 6748,
// JSON-NEXT:            "col": 39,
// JSON-NEXT:            "tokLen": 2
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 6748,
// JSON-NEXT:            "col": 39,
// JSON-NEXT:            "tokLen": 2
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "bool"
// JSON-NEXT:          },
// JSON-NEXT:          "valueCategory": "prvalue",
// JSON-NEXT:          "referencedDecl": {
// JSON-NEXT:           "id": "0x{{.*}}",
// JSON-NEXT:           "kind": "NonTypeTemplateParmDecl",
// JSON-NEXT:           "name": "U3",
// JSON-NEXT:           "type": {
// JSON-NEXT:            "qualType": "bool"
// JSON-NEXT:           }
// JSON-NEXT:          }
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "TemplateTypeParmDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 6688,
// JSON-NEXT:         "line": 187,
// JSON-NEXT:         "col": 19,
// JSON-NEXT:         "tokLen": 2
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 6682,
// JSON-NEXT:          "col": 13,
// JSON-NEXT:          "tokLen": 5
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 6688,
// JSON-NEXT:          "col": 19,
// JSON-NEXT:          "tokLen": 2
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "isReferenced": true,
// JSON-NEXT:        "name": "U1",
// JSON-NEXT:        "tagUsed": "class",
// JSON-NEXT:        "depth": 0,
// JSON-NEXT:        "index": 0
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "NonTypeTemplateParmDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 6697,
// JSON-NEXT:         "col": 28,
// JSON-NEXT:         "tokLen": 2
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 6692,
// JSON-NEXT:          "col": 23,
// JSON-NEXT:          "tokLen": 4
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 6697,
// JSON-NEXT:          "col": 28,
// JSON-NEXT:          "tokLen": 2
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "isReferenced": true,
// JSON-NEXT:        "name": "U2",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "bool"
// JSON-NEXT:        },
// JSON-NEXT:        "depth": 0,
// JSON-NEXT:        "index": 1
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "NonTypeTemplateParmDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 6706,
// JSON-NEXT:         "col": 37,
// JSON-NEXT:         "tokLen": 2
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 6701,
// JSON-NEXT:          "col": 32,
// JSON-NEXT:          "tokLen": 4
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 6706,
// JSON-NEXT:          "col": 37,
// JSON-NEXT:          "tokLen": 2
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "isReferenced": true,
// JSON-NEXT:        "name": "U3",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "bool"
// JSON-NEXT:        },
// JSON-NEXT:        "depth": 0,
// JSON-NEXT:        "index": 2
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "CXXRecordDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 6719,
// JSON-NEXT:         "line": 188,
// JSON-NEXT:         "col": 10,
// JSON-NEXT:         "tokLen": 9
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 6712,
// JSON-NEXT:          "col": 3,
// JSON-NEXT:          "tokLen": 6
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 6719,
// JSON-NEXT:          "col": 10,
// JSON-NEXT:          "tokLen": 9
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "isImplicit": true,
// JSON-NEXT:        "name": "Template2",
// JSON-NEXT:        "tagUsed": "struct"
// JSON-NEXT:       }
// JSON-NEXT:      ]
// JSON-NEXT:     },
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "ClassTemplatePartialSpecializationDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": 8035,
// JSON-NEXT:       "line": 206,
// JSON-NEXT:       "col": 10,
// JSON-NEXT:       "tokLen": 9
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 7985,
// JSON-NEXT:        "line": 205,
// JSON-NEXT:        "col": 3,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 8069,
// JSON-NEXT:        "line": 206,
// JSON-NEXT:        "col": 44,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "name": "Template2",
// JSON-NEXT:      "tagUsed": "struct",
// JSON-NEXT:      "completeDefinition": true,
// JSON-NEXT:      "definitionData": {
// JSON-NEXT:       "canConstDefaultInit": true,
// JSON-NEXT:       "copyAssign": {
// JSON-NEXT:        "hasConstParam": true,
// JSON-NEXT:        "implicitHasConstParam": true,
// JSON-NEXT:        "needsImplicit": true,
// JSON-NEXT:        "simple": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       },
// JSON-NEXT:       "copyCtor": {
// JSON-NEXT:        "hasConstParam": true,
// JSON-NEXT:        "implicitHasConstParam": true,
// JSON-NEXT:        "needsImplicit": true,
// JSON-NEXT:        "simple": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       },
// JSON-NEXT:       "defaultCtor": {
// JSON-NEXT:        "defaultedIsConstexpr": true,
// JSON-NEXT:        "exists": true,
// JSON-NEXT:        "isConstexpr": true,
// JSON-NEXT:        "needsImplicit": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       },
// JSON-NEXT:       "dtor": {
// JSON-NEXT:        "irrelevant": true,
// JSON-NEXT:        "needsImplicit": true,
// JSON-NEXT:        "simple": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       },
// JSON-NEXT:       "hasConstexprNonCopyMoveConstructor": true,
// JSON-NEXT:       "isAggregate": true,
// JSON-NEXT:       "isEmpty": true,
// JSON-NEXT:       "isLiteral": true,
// JSON-NEXT:       "isPOD": true,
// JSON-NEXT:       "isStandardLayout": true,
// JSON-NEXT:       "isTrivial": true,
// JSON-NEXT:       "isTriviallyCopyable": true,
// JSON-NEXT:       "moveAssign": {
// JSON-NEXT:        "exists": true,
// JSON-NEXT:        "needsImplicit": true,
// JSON-NEXT:        "simple": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       },
// JSON-NEXT:       "moveCtor": {
// JSON-NEXT:        "exists": true,
// JSON-NEXT:        "needsImplicit": true,
// JSON-NEXT:        "simple": true,
// JSON-NEXT:        "trivial": true
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "inner": [
// JSON-NEXT:       {
// JSON-NEXT:        "kind": "TemplateArgument",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "TestPartialSpecNTTP::Template1<type-parameter-0-0, value-parameter-0-2>"
// JSON-NEXT:        },
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "TemplateSpecializationType",
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "TestPartialSpecNTTP::Template1<type-parameter-0-0, value-parameter-0-2>"
// JSON-NEXT:          },
// JSON-NEXT:          "isDependent": true,
// JSON-NEXT:          "isInstantiationDependent": true,
// JSON-NEXT:          "templateName": "TestPartialSpecNTTP::Template1",
// JSON-NEXT:          "inner": [
// JSON-NEXT:           {
// JSON-NEXT:            "kind": "TemplateArgument",
// JSON-NEXT:            "type": {
// JSON-NEXT:             "qualType": "type-parameter-0-0"
// JSON-NEXT:            },
// JSON-NEXT:            "inner": [
// JSON-NEXT:             {
// JSON-NEXT:              "id": "0x{{.*}}",
// JSON-NEXT:              "kind": "TemplateTypeParmType",
// JSON-NEXT:              "type": {
// JSON-NEXT:               "qualType": "type-parameter-0-0"
// JSON-NEXT:              },
// JSON-NEXT:              "isDependent": true,
// JSON-NEXT:              "isInstantiationDependent": true,
// JSON-NEXT:              "depth": 0,
// JSON-NEXT:              "index": 0,
// JSON-NEXT:              "decl": {
// JSON-NEXT:               "id": "0x0"
// JSON-NEXT:              }
// JSON-NEXT:             }
// JSON-NEXT:            ]
// JSON-NEXT:           },
// JSON-NEXT:           {
// JSON-NEXT:            "kind": "TemplateArgument",
// JSON-NEXT:            "isExpr": true,
// JSON-NEXT:            "isCanonical": true,
// JSON-NEXT:            "inner": [
// JSON-NEXT:             {
// JSON-NEXT:              "id": "0x{{.*}}",
// JSON-NEXT:              "kind": "DeclRefExpr",
// JSON-NEXT:              "range": {
// JSON-NEXT:               "begin": {
// JSON-NEXT:                "offset": 8059,
// JSON-NEXT:                "col": 34,
// JSON-NEXT:                "tokLen": 2
// JSON-NEXT:               },
// JSON-NEXT:               "end": {
// JSON-NEXT:                "offset": 8059,
// JSON-NEXT:                "col": 34,
// JSON-NEXT:                "tokLen": 2
// JSON-NEXT:               }
// JSON-NEXT:              },
// JSON-NEXT:              "type": {
// JSON-NEXT:               "qualType": "bool"
// JSON-NEXT:              },
// JSON-NEXT:              "valueCategory": "prvalue",
// JSON-NEXT:              "referencedDecl": {
// JSON-NEXT:               "id": "0x{{.*}}",
// JSON-NEXT:               "kind": "NonTypeTemplateParmDecl",
// JSON-NEXT:               "name": "U2",
// JSON-NEXT:               "type": {
// JSON-NEXT:                "qualType": "bool"
// JSON-NEXT:               }
// JSON-NEXT:              }
// JSON-NEXT:             }
// JSON-NEXT:            ]
// JSON-NEXT:           }
// JSON-NEXT:          ]
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "kind": "TemplateArgument",
// JSON-NEXT:        "isExpr": true,
// JSON-NEXT:        "isCanonical": true,
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "DeclRefExpr",
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 8064,
// JSON-NEXT:            "col": 39,
// JSON-NEXT:            "tokLen": 2
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 8064,
// JSON-NEXT:            "col": 39,
// JSON-NEXT:            "tokLen": 2
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "bool"
// JSON-NEXT:          },
// JSON-NEXT:          "valueCategory": "prvalue",
// JSON-NEXT:          "referencedDecl": {
// JSON-NEXT:           "id": "0x{{.*}}",
// JSON-NEXT:           "kind": "NonTypeTemplateParmDecl",
// JSON-NEXT:           "name": "U3",
// JSON-NEXT:           "type": {
// JSON-NEXT:            "qualType": "bool"
// JSON-NEXT:           }
// JSON-NEXT:          }
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "TemplateTypeParmDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 8004,
// JSON-NEXT:         "line": 205,
// JSON-NEXT:         "col": 22,
// JSON-NEXT:         "tokLen": 2
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 7995,
// JSON-NEXT:          "col": 13,
// JSON-NEXT:          "tokLen": 8
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 8004,
// JSON-NEXT:          "col": 22,
// JSON-NEXT:          "tokLen": 2
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "isReferenced": true,
// JSON-NEXT:        "name": "U1",
// JSON-NEXT:        "tagUsed": "typename",
// JSON-NEXT:        "depth": 0,
// JSON-NEXT:        "index": 0
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "NonTypeTemplateParmDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 8013,
// JSON-NEXT:         "col": 31,
// JSON-NEXT:         "tokLen": 2
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 8008,
// JSON-NEXT:          "col": 26,
// JSON-NEXT:          "tokLen": 4
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 8013,
// JSON-NEXT:          "col": 31,
// JSON-NEXT:          "tokLen": 2
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "isReferenced": true,
// JSON-NEXT:        "name": "U3",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "bool"
// JSON-NEXT:        },
// JSON-NEXT:        "depth": 0,
// JSON-NEXT:        "index": 1
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "NonTypeTemplateParmDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 8022,
// JSON-NEXT:         "col": 40,
// JSON-NEXT:         "tokLen": 2
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 8017,
// JSON-NEXT:          "col": 35,
// JSON-NEXT:          "tokLen": 4
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 8022,
// JSON-NEXT:          "col": 40,
// JSON-NEXT:          "tokLen": 2
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "isReferenced": true,
// JSON-NEXT:        "name": "U2",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "bool"
// JSON-NEXT:        },
// JSON-NEXT:        "depth": 0,
// JSON-NEXT:        "index": 2
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "CXXRecordDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 8035,
// JSON-NEXT:         "line": 206,
// JSON-NEXT:         "col": 10,
// JSON-NEXT:         "tokLen": 9
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 8028,
// JSON-NEXT:          "col": 3,
// JSON-NEXT:          "tokLen": 6
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 8035,
// JSON-NEXT:          "col": 10,
// JSON-NEXT:          "tokLen": 9
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "isImplicit": true,
// JSON-NEXT:        "name": "Template2",
// JSON-NEXT:        "tagUsed": "struct"
// JSON-NEXT:       }
// JSON-NEXT:      ]
// JSON-NEXT:     }
// JSON-NEXT:    ]
// JSON-NEXT:   },
// JSON-NEXT:   {
// JSON-NEXT:    "id": "0x{{.*}}",
// JSON-NEXT:    "kind": "NamespaceDecl",
// JSON-NEXT:    "loc": {
// JSON-NEXT:     "offset": 9382,
// JSON-NEXT:     "line": 224,
// JSON-NEXT:     "col": 11,
// JSON-NEXT:     "tokLen": 8
// JSON-NEXT:    },
// JSON-NEXT:    "range": {
// JSON-NEXT:     "begin": {
// JSON-NEXT:      "offset": 9372,
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 9
// JSON-NEXT:     },
// JSON-NEXT:     "end": {
// JSON-NEXT:      "offset": 9791,
// JSON-NEXT:      "line": 238,
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 1
// JSON-NEXT:     }
// JSON-NEXT:    },
// JSON-NEXT:    "name": "GH153540",
// JSON-NEXT:    "inner": [
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "NamespaceDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": 9456,
// JSON-NEXT:       "line": 227,
// JSON-NEXT:       "col": 13,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 9446,
// JSON-NEXT:        "col": 3,
// JSON-NEXT:        "tokLen": 9
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 9507,
// JSON-NEXT:        "line": 229,
// JSON-NEXT:        "col": 3,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "name": "N",
// JSON-NEXT:      "inner": [
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "ClassTemplateDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 9492,
// JSON-NEXT:         "line": 228,
// JSON-NEXT:         "col": 33,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 9464,
// JSON-NEXT:          "col": 5,
// JSON-NEXT:          "tokLen": 8
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 9502,
// JSON-NEXT:          "col": 43,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "name": "S",
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "TemplateTypeParmDecl",
// JSON-NEXT:          "loc": {
// JSON-NEXT:           "offset": 9482,
// JSON-NEXT:           "col": 23,
// JSON-NEXT:           "tokLen": 1
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 9473,
// JSON-NEXT:            "col": 14,
// JSON-NEXT:            "tokLen": 8
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 9482,
// JSON-NEXT:            "col": 23,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "isReferenced": true,
// JSON-NEXT:          "name": "T",
// JSON-NEXT:          "tagUsed": "typename",
// JSON-NEXT:          "depth": 0,
// JSON-NEXT:          "index": 0
// JSON-NEXT:         },
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "CXXRecordDecl",
// JSON-NEXT:          "loc": {
// JSON-NEXT:           "offset": 9492,
// JSON-NEXT:           "col": 33,
// JSON-NEXT:           "tokLen": 1
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 9485,
// JSON-NEXT:            "col": 26,
// JSON-NEXT:            "tokLen": 6
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 9502,
// JSON-NEXT:            "col": 43,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "name": "S",
// JSON-NEXT:          "tagUsed": "struct",
// JSON-NEXT:          "completeDefinition": true,
// JSON-NEXT:          "definitionData": {
// JSON-NEXT:           "canConstDefaultInit": true,
// JSON-NEXT:           "copyAssign": {
// JSON-NEXT:            "hasConstParam": true,
// JSON-NEXT:            "implicitHasConstParam": true,
// JSON-NEXT:            "needsImplicit": true,
// JSON-NEXT:            "simple": true,
// JSON-NEXT:            "trivial": true
// JSON-NEXT:           },
// JSON-NEXT:           "copyCtor": {
// JSON-NEXT:            "hasConstParam": true,
// JSON-NEXT:            "implicitHasConstParam": true,
// JSON-NEXT:            "needsImplicit": true,
// JSON-NEXT:            "simple": true,
// JSON-NEXT:            "trivial": true
// JSON-NEXT:           },
// JSON-NEXT:           "defaultCtor": {
// JSON-NEXT:            "defaultedIsConstexpr": true
// JSON-NEXT:           },
// JSON-NEXT:           "dtor": {
// JSON-NEXT:            "irrelevant": true,
// JSON-NEXT:            "needsImplicit": true,
// JSON-NEXT:            "simple": true,
// JSON-NEXT:            "trivial": true
// JSON-NEXT:           },
// JSON-NEXT:           "hasUserDeclaredConstructor": true,
// JSON-NEXT:           "isEmpty": true,
// JSON-NEXT:           "isStandardLayout": true,
// JSON-NEXT:           "isTriviallyCopyable": true,
// JSON-NEXT:           "moveAssign": {
// JSON-NEXT:            "exists": true,
// JSON-NEXT:            "needsImplicit": true,
// JSON-NEXT:            "simple": true,
// JSON-NEXT:            "trivial": true
// JSON-NEXT:           },
// JSON-NEXT:           "moveCtor": {
// JSON-NEXT:            "exists": true,
// JSON-NEXT:            "needsImplicit": true,
// JSON-NEXT:            "simple": true,
// JSON-NEXT:            "trivial": true
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "inner": [
// JSON-NEXT:           {
// JSON-NEXT:            "id": "0x{{.*}}",
// JSON-NEXT:            "kind": "CXXRecordDecl",
// JSON-NEXT:            "loc": {
// JSON-NEXT:             "offset": 9492,
// JSON-NEXT:             "col": 33,
// JSON-NEXT:             "tokLen": 1
// JSON-NEXT:            },
// JSON-NEXT:            "range": {
// JSON-NEXT:             "begin": {
// JSON-NEXT:              "offset": 9485,
// JSON-NEXT:              "col": 26,
// JSON-NEXT:              "tokLen": 6
// JSON-NEXT:             },
// JSON-NEXT:             "end": {
// JSON-NEXT:              "offset": 9492,
// JSON-NEXT:              "col": 33,
// JSON-NEXT:              "tokLen": 1
// JSON-NEXT:             }
// JSON-NEXT:            },
// JSON-NEXT:            "isImplicit": true,
// JSON-NEXT:            "name": "S",
// JSON-NEXT:            "tagUsed": "struct"
// JSON-NEXT:           },
// JSON-NEXT:           {
// JSON-NEXT:            "id": "0x{{.*}}",
// JSON-NEXT:            "kind": "CXXConstructorDecl",
// JSON-NEXT:            "loc": {
// JSON-NEXT:             "offset": 9496,
// JSON-NEXT:             "col": 37,
// JSON-NEXT:             "tokLen": 1
// JSON-NEXT:            },
// JSON-NEXT:            "range": {
// JSON-NEXT:             "begin": {
// JSON-NEXT:              "offset": 9496,
// JSON-NEXT:              "col": 37,
// JSON-NEXT:              "tokLen": 1
// JSON-NEXT:             },
// JSON-NEXT:             "end": {
// JSON-NEXT:              "offset": 9499,
// JSON-NEXT:              "col": 40,
// JSON-NEXT:              "tokLen": 1
// JSON-NEXT:             }
// JSON-NEXT:            },
// JSON-NEXT:            "name": "S<T>",
// JSON-NEXT:            "type": {
// JSON-NEXT:             "qualType": "void (T)"
// JSON-NEXT:            },
// JSON-NEXT:            "inner": [
// JSON-NEXT:             {
// JSON-NEXT:              "id": "0x{{.*}}",
// JSON-NEXT:              "kind": "ParmVarDecl",
// JSON-NEXT:              "loc": {
// JSON-NEXT:               "offset": 9499,
// JSON-NEXT:               "col": 40,
// JSON-NEXT:               "tokLen": 1
// JSON-NEXT:              },
// JSON-NEXT:              "range": {
// JSON-NEXT:               "begin": {
// JSON-NEXT:                "offset": 9498,
// JSON-NEXT:                "col": 39,
// JSON-NEXT:                "tokLen": 1
// JSON-NEXT:               },
// JSON-NEXT:               "end": {
// JSON-NEXT:                "offset": 9498,
// JSON-NEXT:                "col": 39,
// JSON-NEXT:                "tokLen": 1
// JSON-NEXT:               }
// JSON-NEXT:              },
// JSON-NEXT:              "type": {
// JSON-NEXT:               "qualType": "T"
// JSON-NEXT:              }
// JSON-NEXT:             }
// JSON-NEXT:            ]
// JSON-NEXT:           }
// JSON-NEXT:          ]
// JSON-NEXT:         },
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "ClassTemplateSpecializationDecl",
// JSON-NEXT:          "loc": {
// JSON-NEXT:           "offset": 9492,
// JSON-NEXT:           "col": 33,
// JSON-NEXT:           "tokLen": 1
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 9464,
// JSON-NEXT:            "col": 5,
// JSON-NEXT:            "tokLen": 8
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 9502,
// JSON-NEXT:            "col": 43,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "name": "S",
// JSON-NEXT:          "tagUsed": "struct",
// JSON-NEXT:          "completeDefinition": true,
// JSON-NEXT:          "definitionData": {
// JSON-NEXT:           "canConstDefaultInit": true,
// JSON-NEXT:           "canPassInRegisters": true,
// JSON-NEXT:           "copyAssign": {
// JSON-NEXT:            "hasConstParam": true,
// JSON-NEXT:            "implicitHasConstParam": true,
// JSON-NEXT:            "needsImplicit": true,
// JSON-NEXT:            "simple": true,
// JSON-NEXT:            "trivial": true
// JSON-NEXT:           },
// JSON-NEXT:           "copyCtor": {
// JSON-NEXT:            "hasConstParam": true,
// JSON-NEXT:            "implicitHasConstParam": true,
// JSON-NEXT:            "simple": true,
// JSON-NEXT:            "trivial": true
// JSON-NEXT:           },
// JSON-NEXT:           "defaultCtor": {
// JSON-NEXT:            "defaultedIsConstexpr": true
// JSON-NEXT:           },
// JSON-NEXT:           "dtor": {
// JSON-NEXT:            "irrelevant": true,
// JSON-NEXT:            "simple": true,
// JSON-NEXT:            "trivial": true
// JSON-NEXT:           },
// JSON-NEXT:           "hasUserDeclaredConstructor": true,
// JSON-NEXT:           "isEmpty": true,
// JSON-NEXT:           "isStandardLayout": true,
// JSON-NEXT:           "isTriviallyCopyable": true,
// JSON-NEXT:           "moveAssign": {
// JSON-NEXT:            "exists": true,
// JSON-NEXT:            "needsImplicit": true,
// JSON-NEXT:            "simple": true,
// JSON-NEXT:            "trivial": true
// JSON-NEXT:           },
// JSON-NEXT:           "moveCtor": {
// JSON-NEXT:            "exists": true,
// JSON-NEXT:            "simple": true,
// JSON-NEXT:            "trivial": true
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "inner": [
// JSON-NEXT:           {
// JSON-NEXT:            "kind": "TemplateArgument",
// JSON-NEXT:            "type": {
// JSON-NEXT:             "qualType": "int"
// JSON-NEXT:            },
// JSON-NEXT:            "inner": [
// JSON-NEXT:             {
// JSON-NEXT:              "id": "0x{{.*}}",
// JSON-NEXT:              "kind": "BuiltinType",
// JSON-NEXT:              "type": {
// JSON-NEXT:               "qualType": "int"
// JSON-NEXT:              }
// JSON-NEXT:             }
// JSON-NEXT:            ]
// JSON-NEXT:           },
// JSON-NEXT:           {
// JSON-NEXT:            "id": "0x{{.*}}",
// JSON-NEXT:            "kind": "CXXRecordDecl",
// JSON-NEXT:            "loc": {
// JSON-NEXT:             "offset": 9492,
// JSON-NEXT:             "col": 33,
// JSON-NEXT:             "tokLen": 1
// JSON-NEXT:            },
// JSON-NEXT:            "range": {
// JSON-NEXT:             "begin": {
// JSON-NEXT:              "offset": 9485,
// JSON-NEXT:              "col": 26,
// JSON-NEXT:              "tokLen": 6
// JSON-NEXT:             },
// JSON-NEXT:             "end": {
// JSON-NEXT:              "offset": 9492,
// JSON-NEXT:              "col": 33,
// JSON-NEXT:              "tokLen": 1
// JSON-NEXT:             }
// JSON-NEXT:            },
// JSON-NEXT:            "isImplicit": true,
// JSON-NEXT:            "name": "S",
// JSON-NEXT:            "tagUsed": "struct"
// JSON-NEXT:           },
// JSON-NEXT:           {
// JSON-NEXT:            "id": "0x{{.*}}",
// JSON-NEXT:            "kind": "CXXConstructorDecl",
// JSON-NEXT:            "loc": {
// JSON-NEXT:             "offset": 9496,
// JSON-NEXT:             "col": 37,
// JSON-NEXT:             "tokLen": 1
// JSON-NEXT:            },
// JSON-NEXT:            "range": {
// JSON-NEXT:             "begin": {
// JSON-NEXT:              "offset": 9496,
// JSON-NEXT:              "col": 37,
// JSON-NEXT:              "tokLen": 1
// JSON-NEXT:             },
// JSON-NEXT:             "end": {
// JSON-NEXT:              "offset": 9499,
// JSON-NEXT:              "col": 40,
// JSON-NEXT:              "tokLen": 1
// JSON-NEXT:             }
// JSON-NEXT:            },
// JSON-NEXT:            "isUsed": true,
// JSON-NEXT:            "name": "S",
// JSON-NEXT:            "mangledName": "_ZN8GH1535401N1SIiEC1Ei",
// JSON-NEXT:            "type": {
// JSON-NEXT:             "qualType": "void (int)"
// JSON-NEXT:            },
// JSON-NEXT:            "inner": [
// JSON-NEXT:             {
// JSON-NEXT:              "id": "0x{{.*}}",
// JSON-NEXT:              "kind": "ParmVarDecl",
// JSON-NEXT:              "loc": {
// JSON-NEXT:               "offset": 9499,
// JSON-NEXT:               "col": 40,
// JSON-NEXT:               "tokLen": 1
// JSON-NEXT:              },
// JSON-NEXT:              "range": {
// JSON-NEXT:               "begin": {
// JSON-NEXT:                "offset": 9498,
// JSON-NEXT:                "col": 39,
// JSON-NEXT:                "tokLen": 1
// JSON-NEXT:               },
// JSON-NEXT:               "end": {
// JSON-NEXT:                "offset": 9498,
// JSON-NEXT:                "col": 39,
// JSON-NEXT:                "tokLen": 1
// JSON-NEXT:               }
// JSON-NEXT:              },
// JSON-NEXT:              "type": {
// JSON-NEXT:               "qualType": "int"
// JSON-NEXT:              }
// JSON-NEXT:             }
// JSON-NEXT:            ]
// JSON-NEXT:           },
// JSON-NEXT:           {
// JSON-NEXT:            "id": "0x{{.*}}",
// JSON-NEXT:            "kind": "CXXConstructorDecl",
// JSON-NEXT:            "loc": {
// JSON-NEXT:             "offset": 9492,
// JSON-NEXT:             "col": 33,
// JSON-NEXT:             "tokLen": 1
// JSON-NEXT:            },
// JSON-NEXT:            "range": {
// JSON-NEXT:             "begin": {
// JSON-NEXT:              "offset": 9492,
// JSON-NEXT:              "col": 33,
// JSON-NEXT:              "tokLen": 1
// JSON-NEXT:             },
// JSON-NEXT:             "end": {
// JSON-NEXT:              "offset": 9492,
// JSON-NEXT:              "col": 33,
// JSON-NEXT:              "tokLen": 1
// JSON-NEXT:             }
// JSON-NEXT:            },
// JSON-NEXT:            "isImplicit": true,
// JSON-NEXT:            "name": "S",
// JSON-NEXT:            "mangledName": "_ZN8GH1535401N1SIiEC1ERKS2_",
// JSON-NEXT:            "type": {
// JSON-NEXT:             "qualType": "void (const S<int> &)"
// JSON-NEXT:            },
// JSON-NEXT:            "inline": true,
// JSON-NEXT:            "constexpr": true,
// JSON-NEXT:            "explicitlyDefaulted": "default",
// JSON-NEXT:            "inner": [
// JSON-NEXT:             {
// JSON-NEXT:              "id": "0x{{.*}}",
// JSON-NEXT:              "kind": "ParmVarDecl",
// JSON-NEXT:              "loc": {
// JSON-NEXT:               "offset": 9492,
// JSON-NEXT:               "col": 33,
// JSON-NEXT:               "tokLen": 1
// JSON-NEXT:              },
// JSON-NEXT:              "range": {
// JSON-NEXT:               "begin": {
// JSON-NEXT:                "offset": 9492,
// JSON-NEXT:                "col": 33,
// JSON-NEXT:                "tokLen": 1
// JSON-NEXT:               },
// JSON-NEXT:               "end": {
// JSON-NEXT:                "offset": 9492,
// JSON-NEXT:                "col": 33,
// JSON-NEXT:                "tokLen": 1
// JSON-NEXT:               }
// JSON-NEXT:              },
// JSON-NEXT:              "type": {
// JSON-NEXT:               "qualType": "const S<int> &"
// JSON-NEXT:              }
// JSON-NEXT:             }
// JSON-NEXT:            ]
// JSON-NEXT:           },
// JSON-NEXT:           {
// JSON-NEXT:            "id": "0x{{.*}}",
// JSON-NEXT:            "kind": "CXXConstructorDecl",
// JSON-NEXT:            "loc": {
// JSON-NEXT:             "offset": 9492,
// JSON-NEXT:             "col": 33,
// JSON-NEXT:             "tokLen": 1
// JSON-NEXT:            },
// JSON-NEXT:            "range": {
// JSON-NEXT:             "begin": {
// JSON-NEXT:              "offset": 9492,
// JSON-NEXT:              "col": 33,
// JSON-NEXT:              "tokLen": 1
// JSON-NEXT:             },
// JSON-NEXT:             "end": {
// JSON-NEXT:              "offset": 9492,
// JSON-NEXT:              "col": 33,
// JSON-NEXT:              "tokLen": 1
// JSON-NEXT:             }
// JSON-NEXT:            },
// JSON-NEXT:            "isImplicit": true,
// JSON-NEXT:            "name": "S",
// JSON-NEXT:            "mangledName": "_ZN8GH1535401N1SIiEC1EOS2_",
// JSON-NEXT:            "type": {
// JSON-NEXT:             "qualType": "void (S<int> &&)"
// JSON-NEXT:            },
// JSON-NEXT:            "inline": true,
// JSON-NEXT:            "constexpr": true,
// JSON-NEXT:            "explicitlyDefaulted": "default",
// JSON-NEXT:            "inner": [
// JSON-NEXT:             {
// JSON-NEXT:              "id": "0x{{.*}}",
// JSON-NEXT:              "kind": "ParmVarDecl",
// JSON-NEXT:              "loc": {
// JSON-NEXT:               "offset": 9492,
// JSON-NEXT:               "col": 33,
// JSON-NEXT:               "tokLen": 1
// JSON-NEXT:              },
// JSON-NEXT:              "range": {
// JSON-NEXT:               "begin": {
// JSON-NEXT:                "offset": 9492,
// JSON-NEXT:                "col": 33,
// JSON-NEXT:                "tokLen": 1
// JSON-NEXT:               },
// JSON-NEXT:               "end": {
// JSON-NEXT:                "offset": 9492,
// JSON-NEXT:                "col": 33,
// JSON-NEXT:                "tokLen": 1
// JSON-NEXT:               }
// JSON-NEXT:              },
// JSON-NEXT:              "type": {
// JSON-NEXT:               "qualType": "S<int> &&"
// JSON-NEXT:              }
// JSON-NEXT:             }
// JSON-NEXT:            ]
// JSON-NEXT:           },
// JSON-NEXT:           {
// JSON-NEXT:            "id": "0x{{.*}}",
// JSON-NEXT:            "kind": "CXXDestructorDecl",
// JSON-NEXT:            "loc": {
// JSON-NEXT:             "offset": 9492,
// JSON-NEXT:             "col": 33,
// JSON-NEXT:             "tokLen": 1
// JSON-NEXT:            },
// JSON-NEXT:            "range": {
// JSON-NEXT:             "begin": {
// JSON-NEXT:              "offset": 9492,
// JSON-NEXT:              "col": 33,
// JSON-NEXT:              "tokLen": 1
// JSON-NEXT:             },
// JSON-NEXT:             "end": {
// JSON-NEXT:              "offset": 9492,
// JSON-NEXT:              "col": 33,
// JSON-NEXT:              "tokLen": 1
// JSON-NEXT:             }
// JSON-NEXT:            },
// JSON-NEXT:            "isImplicit": true,
// JSON-NEXT:            "isReferenced": true,
// JSON-NEXT:            "name": "~S",
// JSON-NEXT:            "mangledName": "_ZN8GH1535401N1SIiED1Ev",
// JSON-NEXT:            "type": {
// JSON-NEXT:             "qualType": "void () noexcept"
// JSON-NEXT:            },
// JSON-NEXT:            "inline": true,
// JSON-NEXT:            "constexpr": true,
// JSON-NEXT:            "explicitlyDefaulted": "default"
// JSON-NEXT:           }
// JSON-NEXT:          ]
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "FunctionTemplateDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 9496,
// JSON-NEXT:         "col": 37,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 9464,
// JSON-NEXT:          "col": 5,
// JSON-NEXT:          "tokLen": 8
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 9499,
// JSON-NEXT:          "col": 40,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "isImplicit": true,
// JSON-NEXT:        "name": "<deduction guide for S>",
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "TemplateTypeParmDecl",
// JSON-NEXT:          "loc": {
// JSON-NEXT:           "offset": 9482,
// JSON-NEXT:           "col": 23,
// JSON-NEXT:           "tokLen": 1
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 9473,
// JSON-NEXT:            "col": 14,
// JSON-NEXT:            "tokLen": 8
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 9482,
// JSON-NEXT:            "col": 23,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "isReferenced": true,
// JSON-NEXT:          "name": "T",
// JSON-NEXT:          "tagUsed": "typename",
// JSON-NEXT:          "depth": 0,
// JSON-NEXT:          "index": 0
// JSON-NEXT:         },
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "CXXDeductionGuideDecl",
// JSON-NEXT:          "loc": {
// JSON-NEXT:           "offset": 9496,
// JSON-NEXT:           "col": 37,
// JSON-NEXT:           "tokLen": 1
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 9496,
// JSON-NEXT:            "col": 37,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 9499,
// JSON-NEXT:            "col": 40,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "isImplicit": true,
// JSON-NEXT:          "name": "<deduction guide for S>",
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "auto (T) -> GH153540::N::S<T>"
// JSON-NEXT:          },
// JSON-NEXT:          "inner": [
// JSON-NEXT:           {
// JSON-NEXT:            "id": "0x{{.*}}",
// JSON-NEXT:            "kind": "ParmVarDecl",
// JSON-NEXT:            "loc": {
// JSON-NEXT:             "offset": 9499,
// JSON-NEXT:             "col": 40,
// JSON-NEXT:             "tokLen": 1
// JSON-NEXT:            },
// JSON-NEXT:            "range": {
// JSON-NEXT:             "begin": {
// JSON-NEXT:              "offset": 9498,
// JSON-NEXT:              "col": 39,
// JSON-NEXT:              "tokLen": 1
// JSON-NEXT:             },
// JSON-NEXT:             "end": {
// JSON-NEXT:              "offset": 9498,
// JSON-NEXT:              "col": 39,
// JSON-NEXT:              "tokLen": 1
// JSON-NEXT:             }
// JSON-NEXT:            },
// JSON-NEXT:            "type": {
// JSON-NEXT:             "qualType": "T"
// JSON-NEXT:            }
// JSON-NEXT:           }
// JSON-NEXT:          ]
// JSON-NEXT:         },
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "CXXDeductionGuideDecl",
// JSON-NEXT:          "loc": {
// JSON-NEXT:           "offset": 9496,
// JSON-NEXT:           "col": 37,
// JSON-NEXT:           "tokLen": 1
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 9496,
// JSON-NEXT:            "col": 37,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 9499,
// JSON-NEXT:            "col": 40,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "isImplicit": true,
// JSON-NEXT:          "isUsed": true,
// JSON-NEXT:          "name": "<deduction guide for S>",
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "auto (int) -> GH153540::N::S<int>"
// JSON-NEXT:          },
// JSON-NEXT:          "inner": [
// JSON-NEXT:           {
// JSON-NEXT:            "kind": "TemplateArgument",
// JSON-NEXT:            "type": {
// JSON-NEXT:             "qualType": "int"
// JSON-NEXT:            },
// JSON-NEXT:            "inner": [
// JSON-NEXT:             {
// JSON-NEXT:              "id": "0x{{.*}}",
// JSON-NEXT:              "kind": "BuiltinType",
// JSON-NEXT:              "type": {
// JSON-NEXT:               "qualType": "int"
// JSON-NEXT:              }
// JSON-NEXT:             }
// JSON-NEXT:            ]
// JSON-NEXT:           },
// JSON-NEXT:           {
// JSON-NEXT:            "id": "0x{{.*}}",
// JSON-NEXT:            "kind": "ParmVarDecl",
// JSON-NEXT:            "loc": {
// JSON-NEXT:             "offset": 9499,
// JSON-NEXT:             "col": 40,
// JSON-NEXT:             "tokLen": 1
// JSON-NEXT:            },
// JSON-NEXT:            "range": {
// JSON-NEXT:             "begin": {
// JSON-NEXT:              "offset": 9498,
// JSON-NEXT:              "col": 39,
// JSON-NEXT:              "tokLen": 1
// JSON-NEXT:             },
// JSON-NEXT:             "end": {
// JSON-NEXT:              "offset": 9498,
// JSON-NEXT:              "col": 39,
// JSON-NEXT:              "tokLen": 1
// JSON-NEXT:             }
// JSON-NEXT:            },
// JSON-NEXT:            "type": {
// JSON-NEXT:             "qualType": "int"
// JSON-NEXT:            }
// JSON-NEXT:           }
// JSON-NEXT:          ]
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "FunctionTemplateDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": 9492,
// JSON-NEXT:         "col": 33,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 9464,
// JSON-NEXT:          "col": 5,
// JSON-NEXT:          "tokLen": 8
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 9492,
// JSON-NEXT:          "col": 33,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "isImplicit": true,
// JSON-NEXT:        "name": "<deduction guide for S>",
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "TemplateTypeParmDecl",
// JSON-NEXT:          "loc": {
// JSON-NEXT:           "offset": 9482,
// JSON-NEXT:           "col": 23,
// JSON-NEXT:           "tokLen": 1
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 9473,
// JSON-NEXT:            "col": 14,
// JSON-NEXT:            "tokLen": 8
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 9482,
// JSON-NEXT:            "col": 23,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "isReferenced": true,
// JSON-NEXT:          "name": "T",
// JSON-NEXT:          "tagUsed": "typename",
// JSON-NEXT:          "depth": 0,
// JSON-NEXT:          "index": 0
// JSON-NEXT:         },
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "CXXDeductionGuideDecl",
// JSON-NEXT:          "loc": {
// JSON-NEXT:           "offset": 9492,
// JSON-NEXT:           "col": 33,
// JSON-NEXT:           "tokLen": 1
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 9492,
// JSON-NEXT:            "col": 33,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 9492,
// JSON-NEXT:            "col": 33,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "isImplicit": true,
// JSON-NEXT:          "name": "<deduction guide for S>",
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "auto (GH153540::N::S<T>) -> GH153540::N::S<T>"
// JSON-NEXT:          },
// JSON-NEXT:          "inner": [
// JSON-NEXT:           {
// JSON-NEXT:            "id": "0x{{.*}}",
// JSON-NEXT:            "kind": "ParmVarDecl",
// JSON-NEXT:            "loc": {
// JSON-NEXT:             "offset": 9492,
// JSON-NEXT:             "col": 33,
// JSON-NEXT:             "tokLen": 1
// JSON-NEXT:            },
// JSON-NEXT:            "range": {
// JSON-NEXT:             "begin": {
// JSON-NEXT:              "offset": 9492,
// JSON-NEXT:              "col": 33,
// JSON-NEXT:              "tokLen": 1
// JSON-NEXT:             },
// JSON-NEXT:             "end": {
// JSON-NEXT:              "offset": 9492,
// JSON-NEXT:              "col": 33,
// JSON-NEXT:              "tokLen": 1
// JSON-NEXT:             }
// JSON-NEXT:            },
// JSON-NEXT:            "type": {
// JSON-NEXT:             "qualType": "GH153540::N::S<T>"
// JSON-NEXT:            }
// JSON-NEXT:           }
// JSON-NEXT:          ]
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       }
// JSON-NEXT:      ]
// JSON-NEXT:     },
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "FunctionDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": 9516,
// JSON-NEXT:       "line": 230,
// JSON-NEXT:       "col": 8,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": 9511,
// JSON-NEXT:        "col": 3,
// JSON-NEXT:        "tokLen": 4
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": 9537,
// JSON-NEXT:        "line": 232,
// JSON-NEXT:        "col": 3,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "name": "f",
// JSON-NEXT:      "mangledName": "_ZN8GH1535401fEv",
// JSON-NEXT:      "type": {
// JSON-NEXT:       "qualType": "void ()"
// JSON-NEXT:      },
// JSON-NEXT:      "inner": [
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "CompoundStmt",
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": 9520,
// JSON-NEXT:          "line": 230,
// JSON-NEXT:          "col": 12,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": 9537,
// JSON-NEXT:          "line": 232,
// JSON-NEXT:          "col": 3,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "CXXFunctionalCastExpr",
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": 9526,
// JSON-NEXT:            "line": 231,
// JSON-NEXT:            "col": 5,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": 9532,
// JSON-NEXT:            "col": 11,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "type": {
// JSON-NEXT:           "desugaredQualType": "GH153540::N::S<int>",
// JSON-NEXT:           "qualType": "N::S<int>"
// JSON-NEXT:          },
// JSON-NEXT:          "valueCategory": "prvalue",
// JSON-NEXT:          "castKind": "ConstructorConversion",
// JSON-NEXT:          "conversionFunc": {
// JSON-NEXT:           "id": "0x{{.*}}",
// JSON-NEXT:           "kind": "CXXConstructorDecl",
// JSON-NEXT:           "name": "S",
// JSON-NEXT:           "type": {
// JSON-NEXT:            "qualType": "void (int)"
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "inner": [
// JSON-NEXT:           {
// JSON-NEXT:            "id": "0x{{.*}}",
// JSON-NEXT:            "kind": "CXXConstructExpr",
// JSON-NEXT:            "range": {
// JSON-NEXT:             "begin": {
// JSON-NEXT:              "offset": 9526,
// JSON-NEXT:              "col": 5,
// JSON-NEXT:              "tokLen": 1
// JSON-NEXT:             },
// JSON-NEXT:             "end": {
// JSON-NEXT:              "offset": 9532,
// JSON-NEXT:              "col": 11,
// JSON-NEXT:              "tokLen": 1
// JSON-NEXT:             }
// JSON-NEXT:            },
// JSON-NEXT:            "type": {
// JSON-NEXT:             "desugaredQualType": "GH153540::N::S<int>",
// JSON-NEXT:             "qualType": "N::S<int>"
// JSON-NEXT:            },
// JSON-NEXT:            "valueCategory": "prvalue",
// JSON-NEXT:            "ctorType": {
// JSON-NEXT:             "qualType": "void (int)"
// JSON-NEXT:            },
// JSON-NEXT:            "hadMultipleCandidates": true,
// JSON-NEXT:            "constructionKind": "complete",
// JSON-NEXT:            "inner": [
// JSON-NEXT:             {
// JSON-NEXT:              "id": "0x{{.*}}",
// JSON-NEXT:              "kind": "IntegerLiteral",
// JSON-NEXT:              "range": {
// JSON-NEXT:               "begin": {
// JSON-NEXT:                "offset": 9531,
// JSON-NEXT:                "col": 10,
// JSON-NEXT:                "tokLen": 1
// JSON-NEXT:               },
// JSON-NEXT:               "end": {
// JSON-NEXT:                "offset": 9531,
// JSON-NEXT:                "col": 10,
// JSON-NEXT:                "tokLen": 1
// JSON-NEXT:               }
// JSON-NEXT:              },
// JSON-NEXT:              "type": {
// JSON-NEXT:               "qualType": "int"
// JSON-NEXT:              },
// JSON-NEXT:              "valueCategory": "prvalue",
// JSON-NEXT:              "value": "0"
// JSON-NEXT:             }
// JSON-NEXT:            ]
// JSON-NEXT:           }
// JSON-NEXT:          ]
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       }
// JSON-NEXT:      ]
// JSON-NEXT:     }
// JSON-NEXT:    ]
// JSON-NEXT:   }
// JSON-NEXT:  ]
// JSON-NEXT: }
