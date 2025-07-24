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

// DUMP: UnresolvedLookupExpr {{.*}} '<overloaded function type>' lvalue (ADL) = 'func'
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
}
}

// DUMP: |-NamespaceDecl {{.*}} test6
// DUMP-NEXT: | |-VarTemplateDecl {{.*}} C
// DUMP-NEXT: | | |-TemplateTypeParmDecl {{.*}} class depth 0 index 0 D
// DUMP-NEXT: | | `-VarDecl {{.*}} C 'const bool' constexpr cinit
// DUMP-NEXT: | |   |-value: Int 1
// DUMP-NEXT: | |   |-CXXBoolLiteralExpr {{.*}} 'bool' true
// DUMP-NEXT: | |   `-qualTypeDetail: QualType {{.*}} 'const bool' const
// DUMP-NEXT: | |     `-typeDetails: BuiltinType {{.*}} 'bool'
// DUMP-NEXT: | `-FunctionTemplateDecl {{.*}} func
// DUMP-NEXT: |   |-TemplateTypeParmDecl {{.*}} referenced class depth 0 index 0 Key
// DUMP-NEXT: |   `-FunctionDecl {{.*}} func 'void ()'
// DUMP-NEXT: |     `-CompoundStmt {{.*}} 
// DUMP-NEXT: |       `-UnresolvedLookupExpr {{.*}} '<dependent type>' lvalue (no ADL) = 'C' {{.*}}
// DUMP-NEXT: |         `-TemplateArgument type 'Key':'type-parameter-0-0'
// DUMP-NEXT: |           `-typeDetails: TemplateTypeParmType {{.*}} 'Key' dependent depth 0 index 0
// DUMP-NEXT: |             `-TemplateTypeParm {{.*}} 'Key'

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
} // namespace TestMemberPointerPartialSpec

// DUMP: |-NamespaceDecl {{.*}} TestMemberPointerPartialSpec
// DUMP-NEXT: | |-ClassTemplateDecl {{.*}} A
// DUMP-NEXT: | | |-TemplateTypeParmDecl {{.*}} class depth 0 index 0
// DUMP-NEXT: | | `-CXXRecordDecl {{.*}} struct A
// DUMP-NEXT: | `-ClassTemplatePartialSpecializationDecl {{.*}} struct A explicit_specialization
// DUMP-NEXT: |   |-TemplateArgument type 'type-parameter-0-0 type-parameter-0-1::*'
// DUMP-NEXT: |   | `-typeDetails: MemberPointerType {{.*}} 'type-parameter-0-0 type-parameter-0-1::*' dependent
// DUMP-NEXT: |   |   |-typeDetails: TemplateTypeParmType {{.*}} 'type-parameter-0-1' dependent depth 0 index 1
// DUMP-NEXT: |   |   `-typeDetails: TemplateTypeParmType {{.*}} 'type-parameter-0-0' dependent depth 0 index 0
// DUMP-NEXT: |   |-TemplateTypeParmDecl {{.*}} referenced class depth 0 index 0 T1
// DUMP-NEXT: |   `-TemplateTypeParmDecl {{.*}} class depth 0 index 1 T2

namespace TestDependentMemberPointer {
  template <class U> struct A {
    using X = int U::*;
    using Y = int U::test::*;
    using Z = int U::template V<int>::*;
  };
} // namespace TestDependentMemberPointer

// DUMP: |-NamespaceDecl {{.*}} TestDependentMemberPointer
// DUMP-NEXT: | `-ClassTemplateDecl {{.*}} A
// DUMP-NEXT: |   |-TemplateTypeParmDecl {{.*}} class depth 0 index 0 U
// DUMP-NEXT: |   `-CXXRecordDecl {{.*}} struct A definition
// DUMP-NEXT: |     |-DefinitionData empty aggregate standard_layout trivially_copyable pod trivial literal has_constexpr_non_copy_move_ctor can_const_default_init
// DUMP-NEXT: |     | |-DefaultConstructor exists trivial constexpr needs_implicit defaulted_is_constexpr
// DUMP-NEXT: |     | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT: |     | |-MoveConstructor exists simple trivial needs_implicit
// DUMP-NEXT: |     | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT: |     | |-MoveAssignment exists simple trivial needs_implicit
// DUMP-NEXT: |     | `-Destructor simple irrelevant trivial constexpr needs_implicit
// DUMP-NEXT: |     |-CXXRecordDecl {{.*}} implicit struct A
// DUMP-NEXT: |     |-TypeAliasDecl {{.*}} X 'int U::*'
// DUMP-NEXT: |     | `-typeDetails: MemberPointerType {{.*}} 'int U::*' dependent
// DUMP-NEXT: |     |   |-typeDetails: TemplateTypeParmType {{.*}} 'U' dependent depth 0 index 0
// DUMP-NEXT: |     |   | `-TemplateTypeParm {{.*}} 'U'
// DUMP-NEXT: |     |   `-typeDetails: BuiltinType {{.*}} 'int'
// DUMP-NEXT: |     |-TypeAliasDecl {{.*}} Y 'int U::test::*'
// DUMP-NEXT: |     | `-typeDetails: MemberPointerType {{.*}} 'int U::test::*' dependent
// DUMP-NEXT: |     |   `-typeDetails: BuiltinType {{.*}} 'int'
// DUMP-NEXT: |     `-TypeAliasDecl {{.*}} Z 'int U::template V<int>::*'
// DUMP-NEXT: |       `-typeDetails: MemberPointerType {{.*}} 'int U::template V<int>::*' dependent
// DUMP-NEXT: |         |-typeDetails: DependentTemplateSpecializationType {{.*}} 'template V<int>' dependent
// DUMP-NEXT: |         `-typeDetails: BuiltinType {{.*}} 'int'

namespace TestPartialSpecNTTP {
// DUMP-LABEL: NamespaceDecl {{.+}} TestPartialSpecNTTP{{$}}
  template <class TA1, bool TA2> struct Template1 {};
  template <class TB1, bool TB2> struct Template2 {};

  template <class U1, bool U2, bool U3>
  struct Template2<Template1<U1, U2>, U3> {};

  template <typename U1, bool U3, bool U2>
  struct Template2<Template1<U1, U2>, U3> {};
// CHECK: `-ClassTemplatePartialSpecializationDecl {{.*}} struct Template2 definition explicit_specialization
// CHECK-NEXT:   |-DefinitionData empty aggregate standard_layout trivially_copyable pod trivial literal has_constexpr_non_copy_move_ctor can_const_default_init
// CHECK-NEXT:   | |-DefaultConstructor exists trivial constexpr needs_implicit defaulted_is_constexpr
// CHECK-NEXT:   | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:   | |-MoveConstructor exists simple trivial needs_implicit
// CHECK-NEXT:   | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:   | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT:   | `-Destructor simple irrelevant trivial constexpr needs_implicit
// CHECK-NEXT:   |-TemplateArgument type 'Template1<type-parameter-0-0, value-parameter-0-2>'
// CHECK-NEXT:   | `-typeDetails: TemplateSpecializationType {{.*}} 'Template1<type-parameter-0-0, value-parameter-0-2>' dependent
// CHECK-NEXT:   |   |-name: 'TestPartialSpecNTTP::Template1'
// CHECK-NEXT:   |   | `-ClassTemplateDecl {{.*}} Template1
// CHECK-NEXT:   |   |-TemplateArgument type 'type-parameter-0-0'
// CHECK-NEXT:   |   | `-typeDetails: TemplateTypeParmType {{.*}} 'type-parameter-0-0' dependent depth 0 index 0
// CHECK-NEXT:   |   `-TemplateArgument expr canonical 'value-parameter-0-2'
// CHECK-NEXT:   |     `-DeclRefExpr {{.*}} 'bool' NonTypeTemplateParm {{.*}} 'U2' 'bool'
// CHECK-NEXT:   |-TemplateArgument expr canonical 'value-parameter-0-1'
// CHECK-NEXT:   | `-DeclRefExpr {{.*}} 'bool' NonTypeTemplateParm {{.*}} 'U3' 'bool'
// CHECK-NEXT:   |-TemplateTypeParmDecl {{.*}} referenced typename depth 0 index 0 U1
// CHECK-NEXT:   |-NonTypeTemplateParmDecl {{.*}} referenced 'bool' depth 0 index 1 U3
// CHECK-NEXT:   |-NonTypeTemplateParmDecl {{.*}} referenced 'bool' depth 0 index 2 U2
// CHECK-NEXT:   `-CXXRecordDecl {{.*}} implicit struct Template2

} // namespace TestPartialSpecNTTP

// DUMP:   |-ClassTemplateDecl {{.*}} Template1
// DUMP-NEXT:   | |-TemplateTypeParmDecl {{.*}} class depth 0 index 0 TA1
// DUMP-NEXT:   | |-NonTypeTemplateParmDecl {{.*}} 'bool' depth 0 index 1 TA2
// DUMP-NEXT:   | `-CXXRecordDecl {{.*}} struct Template1 definition
// DUMP-NEXT:   |   |-DefinitionData empty aggregate standard_layout trivially_copyable pod trivial literal has_constexpr_non_copy_move_ctor can_const_default_init
// DUMP-NEXT:   |   | |-DefaultConstructor exists trivial constexpr needs_implicit defaulted_is_constexpr
// DUMP-NEXT:   |   | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:   |   | |-MoveConstructor exists simple trivial needs_implicit
// DUMP-NEXT:   |   | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:   |   | |-MoveAssignment exists simple trivial needs_implicit
// DUMP-NEXT:   |   | `-Destructor simple irrelevant trivial constexpr needs_implicit
// DUMP-NEXT:   |   `-CXXRecordDecl {{.*}} implicit struct Template1
// DUMP-NEXT:   |-ClassTemplateDecl {{.*}} Template2
// DUMP-NEXT:   | |-TemplateTypeParmDecl {{.*}} class depth 0 index 0 TB1
// DUMP-NEXT:   | |-NonTypeTemplateParmDecl {{.*}} 'bool' depth 0 index 1 TB2
// DUMP-NEXT:   | `-CXXRecordDecl {{.*}} struct Template2 definition
// DUMP-NEXT:   |   |-DefinitionData empty aggregate standard_layout trivially_copyable pod trivial literal has_constexpr_non_copy_move_ctor can_const_default_init
// DUMP-NEXT:   |   | |-DefaultConstructor exists trivial constexpr needs_implicit defaulted_is_constexpr
// DUMP-NEXT:   |   | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:   |   | |-MoveConstructor exists simple trivial needs_implicit
// DUMP-NEXT:   |   | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:   |   | |-MoveAssignment exists simple trivial needs_implicit
// DUMP-NEXT:   |   | `-Destructor simple irrelevant trivial constexpr needs_implicit
// DUMP-NEXT:   |   `-CXXRecordDecl {{.*}} implicit struct Template2
// DUMP-NEXT:   |-ClassTemplatePartialSpecializationDecl {{.*}} struct Template2 definition explicit_specialization
// DUMP-NEXT:   | |-DefinitionData empty aggregate standard_layout trivially_copyable pod trivial literal has_constexpr_non_copy_move_ctor can_const_default_init
// DUMP-NEXT:   | | |-DefaultConstructor exists trivial constexpr needs_implicit defaulted_is_constexpr
// DUMP-NEXT:   | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:   | | |-MoveConstructor exists simple trivial needs_implicit
// DUMP-NEXT:   | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:   | | |-MoveAssignment exists simple trivial needs_implicit
// DUMP-NEXT:   | | `-Destructor simple irrelevant trivial constexpr needs_implicit
// DUMP-NEXT:   | |-TemplateArgument type 'Template1<type-parameter-0-0, value-parameter-0-1>'
// DUMP-NEXT:   | | `-typeDetails: TemplateSpecializationType {{.*}} 'Template1<type-parameter-0-0, value-parameter-0-1>' dependent
// DUMP-NEXT:   | |   |-name: 'TestPartialSpecNTTP::Template1'
// DUMP-NEXT:   | |   | `-ClassTemplateDecl {{.*}} Template1
// DUMP-NEXT:   | |   |-TemplateArgument type 'type-parameter-0-0'
// DUMP-NEXT:   | |   | `-typeDetails: TemplateTypeParmType {{.*}} 'type-parameter-0-0' dependent depth 0 index 0
// DUMP-NEXT:   | |   `-TemplateArgument expr canonical 'value-parameter-0-1'
// DUMP-NEXT:   | |     `-DeclRefExpr {{.*}} 'bool' NonTypeTemplateParm {{.*}} 'TA2' 'bool'
// DUMP-NEXT:   | |-TemplateArgument expr canonical 'value-parameter-0-2'
// DUMP-NEXT:   | | `-DeclRefExpr {{.*}} 'bool' NonTypeTemplateParm {{.*}} 'U3' 'bool'
// DUMP-NEXT:   | |-TemplateTypeParmDecl {{.*}} referenced class depth 0 index 0 U1
// DUMP-NEXT:   | |-NonTypeTemplateParmDecl {{.*}} referenced 'bool' depth 0 index 1 U2
// DUMP-NEXT:   | |-NonTypeTemplateParmDecl {{.*}} referenced 'bool' depth 0 index 2 U3
// DUMP-NEXT:   | `-CXXRecordDecl {{.*}} implicit struct Template2
// DUMP-NEXT:   `-ClassTemplatePartialSpecializationDecl {{.*}} struct Template2 definition explicit_specialization
// DUMP-NEXT:     |-DefinitionData empty aggregate standard_layout trivially_copyable pod trivial literal has_constexpr_non_copy_move_ctor can_const_default_init
// DUMP-NEXT:     | |-DefaultConstructor exists trivial constexpr needs_implicit defaulted_is_constexpr
// DUMP-NEXT:     | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:     | |-MoveConstructor exists simple trivial needs_implicit
// DUMP-NEXT:     | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// DUMP-NEXT:     | |-MoveAssignment exists simple trivial needs_implicit
// DUMP-NEXT:     | `-Destructor simple irrelevant trivial constexpr needs_implicit
// DUMP-NEXT:     |-TemplateArgument type 'Template1<type-parameter-0-0, value-parameter-0-2>'
// DUMP-NEXT:     | `-typeDetails: TemplateSpecializationType {{.*}} 'Template1<type-parameter-0-0, value-parameter-0-2>' dependent
// DUMP-NEXT:     |   |-name: 'TestPartialSpecNTTP::Template1'
// DUMP-NEXT:     |   | `-ClassTemplateDecl {{.*}} Template1
// DUMP-NEXT:     |   |-TemplateArgument type 'type-parameter-0-0'
// DUMP-NEXT:     |   | `-typeDetails: TemplateTypeParmType {{.*}} 'type-parameter-0-0' dependent depth 0 index 0
// DUMP-NEXT:     |   `-TemplateArgument expr canonical 'value-parameter-0-2'
// DUMP-NEXT:     |     `-DeclRefExpr {{.*}} 'bool' NonTypeTemplateParm {{.*}} 'U2' 'bool'
// DUMP-NEXT:     |-TemplateArgument expr canonical 'value-parameter-0-1'
// DUMP-NEXT:     | `-DeclRefExpr {{.*}} 'bool' NonTypeTemplateParm {{.*}} 'U3' 'bool'
// DUMP-NEXT:     |-TemplateTypeParmDecl {{.*}} referenced typename depth 0 index 0 U1
// DUMP-NEXT:     |-NonTypeTemplateParmDecl {{.*}} referenced 'bool' depth 0 index 1 U3
// DUMP-NEXT:     |-NonTypeTemplateParmDecl {{.*}} referenced 'bool' depth 0 index 2 U2
// DUMP-NEXT:     `-CXXRecordDecl {{.*}} implicit struct Template2


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
// JSON-NEXT:    "typeDetails": [
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "BuiltinType",
// JSON-NEXT:      "type": {
// JSON-NEXT:       "qualType": "__int128"
// JSON-NEXT:      },
// JSON-NEXT:      "qualDetails": [
// JSON-NEXT:       "signed",
// JSON-NEXT:       "integer"
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
// JSON-NEXT:    "name": "__uint128_t",
// JSON-NEXT:    "type": {
// JSON-NEXT:     "qualType": "unsigned __int128"
// JSON-NEXT:    },
// JSON-NEXT:    "typeDetails": [
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "BuiltinType",
// JSON-NEXT:      "type": {
// JSON-NEXT:       "qualType": "unsigned __int128"
// JSON-NEXT:      },
// JSON-NEXT:      "qualDetails": [
// JSON-NEXT:       "unsigned",
// JSON-NEXT:       "integer"
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
// JSON-NEXT:    "name": "__NSConstantString",
// JSON-NEXT:    "type": {
// JSON-NEXT:     "qualType": "__NSConstantString_tag"
// JSON-NEXT:    },
// JSON-NEXT:    "typeDetails": [
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
// JSON-NEXT:      },
// JSON-NEXT:      "qualDetails": [
// JSON-NEXT:       "struct"
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
// JSON-NEXT:    "name": "__builtin_ms_va_list",
// JSON-NEXT:    "type": {
// JSON-NEXT:     "qualType": "char *"
// JSON-NEXT:    },
// JSON-NEXT:    "typeDetails": [
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "PointerType",
// JSON-NEXT:      "type": {
// JSON-NEXT:       "qualType": "char *"
// JSON-NEXT:      },
// JSON-NEXT:      "qualDetails": [
// JSON-NEXT:       "ptr"
// JSON-NEXT:      ],
// JSON-NEXT:      "typeDetails": [
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "BuiltinType",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "char"
// JSON-NEXT:        },
// JSON-NEXT:        "qualDetails": [
// JSON-NEXT:         "signed",
// JSON-NEXT:         "integer"
// JSON-NEXT:        ]
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
// JSON-NEXT:    "typeDetails": [
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "ConstantArrayType",
// JSON-NEXT:      "type": {
// JSON-NEXT:       "qualType": "__va_list_tag[1]"
// JSON-NEXT:      },
// JSON-NEXT:      "size": 1,
// JSON-NEXT:      "qualDetails": [
// JSON-NEXT:       "array"
// JSON-NEXT:      ],
// JSON-NEXT:      "typeDetails": [
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
// JSON-NEXT:        },
// JSON-NEXT:        "qualDetails": [
// JSON-NEXT:         "struct"
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
// JSON-NEXT:     "offset": {{.*}},
// JSON-NEXT:     "file": "{{.*}}",
// JSON-NEXT:     "line": {{.*}},
// JSON-NEXT:     "col": 8,
// JSON-NEXT:     "tokLen": 3
// JSON-NEXT:    },
// JSON-NEXT:    "range": {
// JSON-NEXT:     "begin": {
// JSON-NEXT:      "offset": {{.*}},
// JSON-NEXT:      "line": {{.*}},
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 8
// JSON-NEXT:     },
// JSON-NEXT:     "end": {
// JSON-NEXT:      "offset": {{.*}},
// JSON-NEXT:      "line": {{.*}},
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
// JSON-NEXT:       "offset": {{.*}},
// JSON-NEXT:       "line": {{.*}},
// JSON-NEXT:       "col": 15,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "col": 11,
// JSON-NEXT:        "tokLen": 3
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": {{.*}},
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
// JSON-NEXT:       "offset": {{.*}},
// JSON-NEXT:       "col": 27,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "col": 18,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": {{.*}},
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
// JSON-NEXT:       "offset": {{.*}},
// JSON-NEXT:       "col": 34,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "col": 30,
// JSON-NEXT:        "tokLen": 3
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": {{.*}},
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
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 38,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 38,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
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
// JSON-NEXT:       "offset": {{.*}},
// JSON-NEXT:       "line": {{.*}},
// JSON-NEXT:       "col": 8,
// JSON-NEXT:       "tokLen": 3
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 6
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "line": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "line": {{.*}},
// JSON-NEXT:         "col": 8,
// JSON-NEXT:         "tokLen": 3
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 1,
// JSON-NEXT:          "tokLen": 6
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 8,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "isImplicit": true,
// JSON-NEXT:        "isReferenced": true,
// JSON-NEXT:        "name": "foo",
// JSON-NEXT:        "tagUsed": "struct"
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "FieldDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "line": {{.*}},
// JSON-NEXT:         "col": 7,
// JSON-NEXT:         "tokLen": 8
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 3,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 7,
// JSON-NEXT:          "tokLen": 8
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "name": "constant",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "int"
// JSON-NEXT:        },
// JSON-NEXT:        "qualType": {
// JSON-NEXT:         "id": "0x{{.*}}",
// JSON-NEXT:         "kind": "QualType",
// JSON-NEXT:         "type": {
// JSON-NEXT:          "qualType": "int"
// JSON-NEXT:         },
// JSON-NEXT:         "qualifiers": "",
// JSON-NEXT:         "qualDetails": [
// JSON-NEXT:          "signed",
// JSON-NEXT:          "integer"
// JSON-NEXT:         ]
// JSON-NEXT:        }
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "CXXConstructorDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "line": {{.*}},
// JSON-NEXT:         "col": 3,
// JSON-NEXT:         "tokLen": 3
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 3,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 9,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "line": {{.*}},
// JSON-NEXT:         "col": 5,
// JSON-NEXT:         "tokLen": 6
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 3,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 14,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
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
// JSON-NEXT:              "offset": {{.*}},
// JSON-NEXT:              "col": 16,
// JSON-NEXT:              "tokLen": 6
// JSON-NEXT:             },
// JSON-NEXT:             "end": {
// JSON-NEXT:              "offset": {{.*}},
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
// JSON-NEXT:                "offset": {{.*}},
// JSON-NEXT:                "col": 23,
// JSON-NEXT:                "tokLen": 1
// JSON-NEXT:               },
// JSON-NEXT:               "end": {
// JSON-NEXT:                "offset": {{.*}},
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
// JSON-NEXT:                  "offset": {{.*}},
// JSON-NEXT:                  "col": 25,
// JSON-NEXT:                  "tokLen": 1
// JSON-NEXT:                 },
// JSON-NEXT:                 "end": {
// JSON-NEXT:                  "offset": {{.*}},
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
// JSON-NEXT:                    "offset": {{.*}},
// JSON-NEXT:                    "col": 25,
// JSON-NEXT:                    "tokLen": 1
// JSON-NEXT:                   },
// JSON-NEXT:                   "end": {
// JSON-NEXT:                    "offset": {{.*}},
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
// JSON-NEXT:                  },
// JSON-NEXT:                  "qualType": {
// JSON-NEXT:                   "refId": "0x{{.*}}",
// JSON-NEXT:                   "qualDetails": [
// JSON-NEXT:                    "signed",
// JSON-NEXT:                    "integer"
// JSON-NEXT:                   ]
// JSON-NEXT:                  }
// JSON-NEXT:                 },
// JSON-NEXT:                 {
// JSON-NEXT:                  "id": "0x{{.*}}",
// JSON-NEXT:                  "kind": "DeclRefExpr",
// JSON-NEXT:                  "range": {
// JSON-NEXT:                   "begin": {
// JSON-NEXT:                    "offset": {{.*}},
// JSON-NEXT:                    "col": 29,
// JSON-NEXT:                    "tokLen": 1
// JSON-NEXT:                   },
// JSON-NEXT:                   "end": {
// JSON-NEXT:                    "offset": {{.*}},
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
// JSON-NEXT:                  },
// JSON-NEXT:                  "qualType": {
// JSON-NEXT:                   "refId": "0x{{.*}}",
// JSON-NEXT:                   "qualDetails": [
// JSON-NEXT:                    "signed",
// JSON-NEXT:                    "integer"
// JSON-NEXT:                   ]
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
// JSON-NEXT:       "offset": {{.*}},
// JSON-NEXT:       "line": {{.*}},
// JSON-NEXT:       "col": 8,
// JSON-NEXT:       "tokLen": 3
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "line": {{.*}},
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "line": {{.*}},
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
// JSON-NEXT:        "typeDetails": [
// JSON-NEXT:         {
// JSON-NEXT:          "qualDetails": [
// JSON-NEXT:           "signed",
// JSON-NEXT:           "integer"
// JSON-NEXT:          ],
// JSON-NEXT:          "typeDetails": [
// JSON-NEXT:           {
// JSON-NEXT:            "refId": "0x{{.*}}"
// JSON-NEXT:           }
// JSON-NEXT:          ]
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "line": {{.*}},
// JSON-NEXT:         "col": 8,
// JSON-NEXT:         "tokLen": 3
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 1,
// JSON-NEXT:          "tokLen": 6
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "line": {{.*}},
// JSON-NEXT:         "col": 7,
// JSON-NEXT:         "tokLen": 8
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 3,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 7,
// JSON-NEXT:          "tokLen": 8
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "name": "constant",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "int"
// JSON-NEXT:        },
// JSON-NEXT:        "qualType": {
// JSON-NEXT:         "refId": "0x{{.*}}",
// JSON-NEXT:         "qualDetails": [
// JSON-NEXT:          "signed",
// JSON-NEXT:          "integer"
// JSON-NEXT:         ]
// JSON-NEXT:        }
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "CXXConstructorDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "line": {{.*}},
// JSON-NEXT:         "col": 3,
// JSON-NEXT:         "tokLen": 3
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 3,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 9,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "line": {{.*}},
// JSON-NEXT:         "col": 5,
// JSON-NEXT:         "tokLen": 6
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 3,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 14,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
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
// JSON-NEXT:              "offset": {{.*}},
// JSON-NEXT:              "col": 16,
// JSON-NEXT:              "tokLen": 6
// JSON-NEXT:             },
// JSON-NEXT:             "end": {
// JSON-NEXT:              "offset": {{.*}},
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
// JSON-NEXT:                "offset": {{.*}},
// JSON-NEXT:                "col": 23,
// JSON-NEXT:                "tokLen": 1
// JSON-NEXT:               },
// JSON-NEXT:               "end": {
// JSON-NEXT:                "offset": {{.*}},
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
// JSON-NEXT:                  "offset": {{.*}},
// JSON-NEXT:                  "col": 25,
// JSON-NEXT:                  "tokLen": 1
// JSON-NEXT:                 },
// JSON-NEXT:                 "end": {
// JSON-NEXT:                  "offset": {{.*}},
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
// JSON-NEXT:                    "offset": {{.*}},
// JSON-NEXT:                    "col": 25,
// JSON-NEXT:                    "tokLen": 1
// JSON-NEXT:                   },
// JSON-NEXT:                   "end": {
// JSON-NEXT:                    "offset": {{.*}},
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
// JSON-NEXT:                     "offset": {{.*}},
// JSON-NEXT:                     "line": {{.*}},
// JSON-NEXT:                     "col": 15,
// JSON-NEXT:                     "tokLen": 1
// JSON-NEXT:                    },
// JSON-NEXT:                    "range": {
// JSON-NEXT:                     "begin": {
// JSON-NEXT:                      "offset": {{.*}},
// JSON-NEXT:                      "col": 11,
// JSON-NEXT:                      "tokLen": 3
// JSON-NEXT:                     },
// JSON-NEXT:                     "end": {
// JSON-NEXT:                      "offset": {{.*}},
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
// JSON-NEXT:                      "offset": {{.*}},
// JSON-NEXT:                      "line": {{.*}},
// JSON-NEXT:                      "col": 25,
// JSON-NEXT:                      "tokLen": 1
// JSON-NEXT:                     },
// JSON-NEXT:                     "end": {
// JSON-NEXT:                      "offset": {{.*}},
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
// JSON-NEXT:                    "offset": {{.*}},
// JSON-NEXT:                    "col": 29,
// JSON-NEXT:                    "tokLen": 1
// JSON-NEXT:                   },
// JSON-NEXT:                   "end": {
// JSON-NEXT:                    "offset": {{.*}},
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
// JSON-NEXT:                     "offset": {{.*}},
// JSON-NEXT:                     "line": {{.*}},
// JSON-NEXT:                     "col": 34,
// JSON-NEXT:                     "tokLen": 1
// JSON-NEXT:                    },
// JSON-NEXT:                    "range": {
// JSON-NEXT:                     "begin": {
// JSON-NEXT:                      "offset": {{.*}},
// JSON-NEXT:                      "col": 30,
// JSON-NEXT:                      "tokLen": 3
// JSON-NEXT:                     },
// JSON-NEXT:                     "end": {
// JSON-NEXT:                      "offset": {{.*}},
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
// JSON-NEXT:                        "offset": {{.*}},
// JSON-NEXT:                        "col": 38,
// JSON-NEXT:                        "tokLen": 1
// JSON-NEXT:                       },
// JSON-NEXT:                       "end": {
// JSON-NEXT:                        "offset": {{.*}},
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
// JSON-NEXT:                          "offset": {{.*}},
// JSON-NEXT:                          "col": 38,
// JSON-NEXT:                          "tokLen": 1
// JSON-NEXT:                         },
// JSON-NEXT:                         "end": {
// JSON-NEXT:                          "offset": {{.*}},
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
// JSON-NEXT:                      "offset": {{.*}},
// JSON-NEXT:                      "line": {{.*}},
// JSON-NEXT:                      "col": 29,
// JSON-NEXT:                      "tokLen": 1
// JSON-NEXT:                     },
// JSON-NEXT:                     "end": {
// JSON-NEXT:                      "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "line": {{.*}},
// JSON-NEXT:         "col": 8,
// JSON-NEXT:         "tokLen": 3
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 8,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:           "offset": {{.*}},
// JSON-NEXT:           "col": 8,
// JSON-NEXT:           "tokLen": 3
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 8,
// JSON-NEXT:            "tokLen": 3
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 8,
// JSON-NEXT:            "tokLen": 3
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "const foo<5, int> &"
// JSON-NEXT:          },
// JSON-NEXT:          "typeDetails": [
// JSON-NEXT:           {
// JSON-NEXT:            "id": "0x{{.*}}",
// JSON-NEXT:            "kind": "LValueReferenceType",
// JSON-NEXT:            "type": {
// JSON-NEXT:             "qualType": "const foo<5, int> &"
// JSON-NEXT:            },
// JSON-NEXT:            "qualDetails": [],
// JSON-NEXT:            "qualTypeDetail": [
// JSON-NEXT:             {
// JSON-NEXT:              "qualType": {
// JSON-NEXT:               "id": "0x{{.*}}",
// JSON-NEXT:               "kind": "QualType",
// JSON-NEXT:               "type": {
// JSON-NEXT:                "qualType": "const foo<5, int>"
// JSON-NEXT:               },
// JSON-NEXT:               "qualifiers": "const",
// JSON-NEXT:               "qualDetails": [
// JSON-NEXT:                "struct"
// JSON-NEXT:               ]
// JSON-NEXT:              },
// JSON-NEXT:              "typeDetails": [
// JSON-NEXT:               {
// JSON-NEXT:                "id": "0x{{.*}}",
// JSON-NEXT:                "kind": "ElaboratedType",
// JSON-NEXT:                "type": {
// JSON-NEXT:                 "qualType": "foo<5, int>"
// JSON-NEXT:                },
// JSON-NEXT:                "qualDetails": [
// JSON-NEXT:                 "struct"
// JSON-NEXT:                ],
// JSON-NEXT:                "typeDetails": [
// JSON-NEXT:                 {
// JSON-NEXT:                  "id": "0x{{.*}}",
// JSON-NEXT:                  "kind": "RecordType",
// JSON-NEXT:                  "type": {
// JSON-NEXT:                   "qualType": "foo<5, int>"
// JSON-NEXT:                  },
// JSON-NEXT:                  "decl": {
// JSON-NEXT:                   "id": "0x{{.*}}",
// JSON-NEXT:                   "kind": "ClassTemplateSpecializationDecl",
// JSON-NEXT:                   "name": "foo"
// JSON-NEXT:                  },
// JSON-NEXT:                  "qualDetails": [
// JSON-NEXT:                   "struct"
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "col": 8,
// JSON-NEXT:         "tokLen": 3
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 8,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:           "offset": {{.*}},
// JSON-NEXT:           "col": 8,
// JSON-NEXT:           "tokLen": 3
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 8,
// JSON-NEXT:            "tokLen": 3
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 8,
// JSON-NEXT:            "tokLen": 3
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "foo<5, int> &&"
// JSON-NEXT:          },
// JSON-NEXT:          "typeDetails": [
// JSON-NEXT:           {
// JSON-NEXT:            "id": "0x{{.*}}",
// JSON-NEXT:            "kind": "RValueReferenceType",
// JSON-NEXT:            "type": {
// JSON-NEXT:             "qualType": "foo<5, int> &&"
// JSON-NEXT:            },
// JSON-NEXT:            "qualDetails": [],
// JSON-NEXT:            "typeDetails": [
// JSON-NEXT:             {
// JSON-NEXT:              "qualDetails": [
// JSON-NEXT:               "struct"
// JSON-NEXT:              ],
// JSON-NEXT:              "typeDetails": [
// JSON-NEXT:               {
// JSON-NEXT:                "refId": "0x{{.*}}"
// JSON-NEXT:               },
// JSON-NEXT:               {
// JSON-NEXT:                "decl": {
// JSON-NEXT:                 "id": "0x{{.*}}",
// JSON-NEXT:                 "kind": "ClassTemplateSpecializationDecl",
// JSON-NEXT:                 "name": "foo"
// JSON-NEXT:                },
// JSON-NEXT:                "qualDetails": [
// JSON-NEXT:                 "struct"
// JSON-NEXT:                ],
// JSON-NEXT:                "typeDetails": [
// JSON-NEXT:                 {
// JSON-NEXT:                  "refId": "0x{{.*}}"
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
// JSON-NEXT:        "kind": "CXXDestructorDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "col": 8,
// JSON-NEXT:         "tokLen": 3
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 8,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:       "offset": {{.*}},
// JSON-NEXT:       "col": 8,
// JSON-NEXT:       "tokLen": 3
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "line": {{.*}},
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "line": {{.*}},
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
// JSON-NEXT:        "typeDetails": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "BuiltinType",
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "double"
// JSON-NEXT:          },
// JSON-NEXT:          "qualDetails": [
// JSON-NEXT:           "fpp"
// JSON-NEXT:          ]
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "line": {{.*}},
// JSON-NEXT:         "col": 8,
// JSON-NEXT:         "tokLen": 3
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 1,
// JSON-NEXT:          "tokLen": 6
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "line": {{.*}},
// JSON-NEXT:         "col": 7,
// JSON-NEXT:         "tokLen": 8
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 3,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 7,
// JSON-NEXT:          "tokLen": 8
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "name": "constant",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "int"
// JSON-NEXT:        },
// JSON-NEXT:        "qualType": {
// JSON-NEXT:         "refId": "0x{{.*}}",
// JSON-NEXT:         "qualDetails": [
// JSON-NEXT:          "signed",
// JSON-NEXT:          "integer"
// JSON-NEXT:         ]
// JSON-NEXT:        }
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "CXXConstructorDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "line": {{.*}},
// JSON-NEXT:         "col": 3,
// JSON-NEXT:         "tokLen": 3
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 3,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 9,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "line": {{.*}},
// JSON-NEXT:         "col": 5,
// JSON-NEXT:         "tokLen": 6
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 3,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 14,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
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
// JSON-NEXT:              "offset": {{.*}},
// JSON-NEXT:              "col": 16,
// JSON-NEXT:              "tokLen": 6
// JSON-NEXT:             },
// JSON-NEXT:             "end": {
// JSON-NEXT:              "offset": {{.*}},
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
// JSON-NEXT:                "offset": {{.*}},
// JSON-NEXT:                "col": 23,
// JSON-NEXT:                "tokLen": 1
// JSON-NEXT:               },
// JSON-NEXT:               "end": {
// JSON-NEXT:                "offset": {{.*}},
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
// JSON-NEXT:                  "offset": {{.*}},
// JSON-NEXT:                  "col": 25,
// JSON-NEXT:                  "tokLen": 1
// JSON-NEXT:                 },
// JSON-NEXT:                 "end": {
// JSON-NEXT:                  "offset": {{.*}},
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
// JSON-NEXT:                    "offset": {{.*}},
// JSON-NEXT:                    "col": 25,
// JSON-NEXT:                    "tokLen": 1
// JSON-NEXT:                   },
// JSON-NEXT:                   "end": {
// JSON-NEXT:                    "offset": {{.*}},
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
// JSON-NEXT:                      "offset": {{.*}},
// JSON-NEXT:                      "col": 25,
// JSON-NEXT:                      "tokLen": 1
// JSON-NEXT:                     },
// JSON-NEXT:                     "end": {
// JSON-NEXT:                      "offset": {{.*}},
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
// JSON-NEXT:                       "offset": {{.*}},
// JSON-NEXT:                       "line": {{.*}},
// JSON-NEXT:                       "col": 15,
// JSON-NEXT:                       "tokLen": 1
// JSON-NEXT:                      },
// JSON-NEXT:                      "range": {
// JSON-NEXT:                       "begin": {
// JSON-NEXT:                        "offset": {{.*}},
// JSON-NEXT:                        "col": 11,
// JSON-NEXT:                        "tokLen": 3
// JSON-NEXT:                       },
// JSON-NEXT:                       "end": {
// JSON-NEXT:                        "offset": {{.*}},
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
// JSON-NEXT:                        "offset": {{.*}},
// JSON-NEXT:                        "line": {{.*}},
// JSON-NEXT:                        "col": 25,
// JSON-NEXT:                        "tokLen": 1
// JSON-NEXT:                       },
// JSON-NEXT:                       "end": {
// JSON-NEXT:                        "offset": {{.*}},
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
// JSON-NEXT:                      "offset": {{.*}},
// JSON-NEXT:                      "col": 29,
// JSON-NEXT:                      "tokLen": 1
// JSON-NEXT:                     },
// JSON-NEXT:                     "end": {
// JSON-NEXT:                      "offset": {{.*}},
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
// JSON-NEXT:                       "offset": {{.*}},
// JSON-NEXT:                       "line": {{.*}},
// JSON-NEXT:                       "col": 34,
// JSON-NEXT:                       "tokLen": 1
// JSON-NEXT:                      },
// JSON-NEXT:                      "range": {
// JSON-NEXT:                       "begin": {
// JSON-NEXT:                        "offset": {{.*}},
// JSON-NEXT:                        "col": 30,
// JSON-NEXT:                        "tokLen": 3
// JSON-NEXT:                       },
// JSON-NEXT:                       "end": {
// JSON-NEXT:                        "offset": {{.*}},
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
// JSON-NEXT:                          "offset": {{.*}},
// JSON-NEXT:                          "col": 38,
// JSON-NEXT:                          "tokLen": 1
// JSON-NEXT:                         },
// JSON-NEXT:                         "end": {
// JSON-NEXT:                          "offset": {{.*}},
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
// JSON-NEXT:                            "offset": {{.*}},
// JSON-NEXT:                            "col": 38,
// JSON-NEXT:                            "tokLen": 1
// JSON-NEXT:                           },
// JSON-NEXT:                           "end": {
// JSON-NEXT:                            "offset": {{.*}},
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
// JSON-NEXT:                        "offset": {{.*}},
// JSON-NEXT:                        "line": {{.*}},
// JSON-NEXT:                        "col": 29,
// JSON-NEXT:                        "tokLen": 1
// JSON-NEXT:                       },
// JSON-NEXT:                       "end": {
// JSON-NEXT:                        "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "line": {{.*}},
// JSON-NEXT:         "col": 8,
// JSON-NEXT:         "tokLen": 3
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 8,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:           "offset": {{.*}},
// JSON-NEXT:           "col": 8,
// JSON-NEXT:           "tokLen": 3
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 8,
// JSON-NEXT:            "tokLen": 3
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 8,
// JSON-NEXT:            "tokLen": 3
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "const foo<2, double, 3> &"
// JSON-NEXT:          },
// JSON-NEXT:          "typeDetails": [
// JSON-NEXT:           {
// JSON-NEXT:            "id": "0x{{.*}}",
// JSON-NEXT:            "kind": "LValueReferenceType",
// JSON-NEXT:            "type": {
// JSON-NEXT:             "qualType": "const foo<2, double, 3> &"
// JSON-NEXT:            },
// JSON-NEXT:            "qualDetails": [],
// JSON-NEXT:            "qualTypeDetail": [
// JSON-NEXT:             {
// JSON-NEXT:              "qualType": {
// JSON-NEXT:               "id": "0x{{.*}}",
// JSON-NEXT:               "kind": "QualType",
// JSON-NEXT:               "type": {
// JSON-NEXT:                "qualType": "const foo<2, double, 3>"
// JSON-NEXT:               },
// JSON-NEXT:               "qualifiers": "const",
// JSON-NEXT:               "qualDetails": [
// JSON-NEXT:                "struct"
// JSON-NEXT:               ]
// JSON-NEXT:              },
// JSON-NEXT:              "typeDetails": [
// JSON-NEXT:               {
// JSON-NEXT:                "id": "0x{{.*}}",
// JSON-NEXT:                "kind": "ElaboratedType",
// JSON-NEXT:                "type": {
// JSON-NEXT:                 "qualType": "foo<2, double, 3>"
// JSON-NEXT:                },
// JSON-NEXT:                "qualDetails": [
// JSON-NEXT:                 "struct"
// JSON-NEXT:                ],
// JSON-NEXT:                "typeDetails": [
// JSON-NEXT:                 {
// JSON-NEXT:                  "id": "0x{{.*}}",
// JSON-NEXT:                  "kind": "RecordType",
// JSON-NEXT:                  "type": {
// JSON-NEXT:                   "qualType": "foo<2, double, 3>"
// JSON-NEXT:                  },
// JSON-NEXT:                  "decl": {
// JSON-NEXT:                   "id": "0x{{.*}}",
// JSON-NEXT:                   "kind": "ClassTemplateSpecializationDecl",
// JSON-NEXT:                   "name": "foo"
// JSON-NEXT:                  },
// JSON-NEXT:                  "qualDetails": [
// JSON-NEXT:                   "struct"
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "col": 8,
// JSON-NEXT:         "tokLen": 3
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 8,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:           "offset": {{.*}},
// JSON-NEXT:           "col": 8,
// JSON-NEXT:           "tokLen": 3
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 8,
// JSON-NEXT:            "tokLen": 3
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 8,
// JSON-NEXT:            "tokLen": 3
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "foo<2, double, 3> &&"
// JSON-NEXT:          },
// JSON-NEXT:          "typeDetails": [
// JSON-NEXT:           {
// JSON-NEXT:            "id": "0x{{.*}}",
// JSON-NEXT:            "kind": "RValueReferenceType",
// JSON-NEXT:            "type": {
// JSON-NEXT:             "qualType": "foo<2, double, 3> &&"
// JSON-NEXT:            },
// JSON-NEXT:            "qualDetails": [],
// JSON-NEXT:            "typeDetails": [
// JSON-NEXT:             {
// JSON-NEXT:              "qualDetails": [
// JSON-NEXT:               "struct"
// JSON-NEXT:              ],
// JSON-NEXT:              "typeDetails": [
// JSON-NEXT:               {
// JSON-NEXT:                "refId": "0x{{.*}}"
// JSON-NEXT:               },
// JSON-NEXT:               {
// JSON-NEXT:                "decl": {
// JSON-NEXT:                 "id": "0x{{.*}}",
// JSON-NEXT:                 "kind": "ClassTemplateSpecializationDecl",
// JSON-NEXT:                 "name": "foo"
// JSON-NEXT:                },
// JSON-NEXT:                "qualDetails": [
// JSON-NEXT:                 "struct"
// JSON-NEXT:                ],
// JSON-NEXT:                "typeDetails": [
// JSON-NEXT:                 {
// JSON-NEXT:                  "refId": "0x{{.*}}"
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
// JSON-NEXT:        "kind": "CXXDestructorDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "col": 8,
// JSON-NEXT:         "tokLen": 3
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 8,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:     "offset": {{.*}},
// JSON-NEXT:     "line": {{.*}},
// JSON-NEXT:     "col": 3,
// JSON-NEXT:     "tokLen": 3
// JSON-NEXT:    },
// JSON-NEXT:    "range": {
// JSON-NEXT:     "begin": {
// JSON-NEXT:      "offset": {{.*}},
// JSON-NEXT:      "line": {{.*}},
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 8
// JSON-NEXT:     },
// JSON-NEXT:     "end": {
// JSON-NEXT:      "offset": {{.*}},
// JSON-NEXT:      "line": {{.*}},
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
// JSON-NEXT:       "offset": {{.*}},
// JSON-NEXT:       "line": {{.*}},
// JSON-NEXT:       "col": 15,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "col": 11,
// JSON-NEXT:        "tokLen": 3
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": {{.*}},
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
// JSON-NEXT:       "offset": {{.*}},
// JSON-NEXT:       "col": 27,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "col": 18,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": {{.*}},
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
// JSON-NEXT:       "offset": {{.*}},
// JSON-NEXT:       "line": {{.*}},
// JSON-NEXT:       "col": 3,
// JSON-NEXT:       "tokLen": 3
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "line": {{.*}},
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
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "line": {{.*}},
// JSON-NEXT:          "col": 9,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "line": {{.*}},
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
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "line": {{.*}},
// JSON-NEXT:            "col": 3,
// JSON-NEXT:            "tokLen": 6
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
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
// JSON-NEXT:              "offset": {{.*}},
// JSON-NEXT:              "col": 10,
// JSON-NEXT:              "tokLen": 1
// JSON-NEXT:             },
// JSON-NEXT:             "end": {
// JSON-NEXT:              "offset": {{.*}},
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
// JSON-NEXT:                "offset": {{.*}},
// JSON-NEXT:                "col": 12,
// JSON-NEXT:                "tokLen": 1
// JSON-NEXT:               },
// JSON-NEXT:               "end": {
// JSON-NEXT:                "offset": {{.*}},
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
// JSON-NEXT:              },
// JSON-NEXT:              "qualType": {
// JSON-NEXT:               "refId": "0x{{.*}}",
// JSON-NEXT:               "qualDetails": [
// JSON-NEXT:                "signed",
// JSON-NEXT:                "integer"
// JSON-NEXT:               ]
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
// JSON-NEXT:       "offset": {{.*}},
// JSON-NEXT:       "line": {{.*}},
// JSON-NEXT:       "col": 3,
// JSON-NEXT:       "tokLen": 3
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "line": {{.*}},
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
// JSON-NEXT:        "typeDetails": [
// JSON-NEXT:         {
// JSON-NEXT:          "qualDetails": [
// JSON-NEXT:           "signed",
// JSON-NEXT:           "integer"
// JSON-NEXT:          ],
// JSON-NEXT:          "typeDetails": [
// JSON-NEXT:           {
// JSON-NEXT:            "refId": "0x{{.*}}"
// JSON-NEXT:           }
// JSON-NEXT:          ]
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "CompoundStmt",
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "line": {{.*}},
// JSON-NEXT:          "col": 9,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "line": {{.*}},
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
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "line": {{.*}},
// JSON-NEXT:            "col": 3,
// JSON-NEXT:            "tokLen": 6
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
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
// JSON-NEXT:              "offset": {{.*}},
// JSON-NEXT:              "col": 10,
// JSON-NEXT:              "tokLen": 1
// JSON-NEXT:             },
// JSON-NEXT:             "end": {
// JSON-NEXT:              "offset": {{.*}},
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
// JSON-NEXT:                "offset": {{.*}},
// JSON-NEXT:                "col": 12,
// JSON-NEXT:                "tokLen": 1
// JSON-NEXT:               },
// JSON-NEXT:               "end": {
// JSON-NEXT:                "offset": {{.*}},
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
// JSON-NEXT:                 "offset": {{.*}},
// JSON-NEXT:                 "line": {{.*}},
// JSON-NEXT:                 "col": 15,
// JSON-NEXT:                 "tokLen": 1
// JSON-NEXT:                },
// JSON-NEXT:                "range": {
// JSON-NEXT:                 "begin": {
// JSON-NEXT:                  "offset": {{.*}},
// JSON-NEXT:                  "col": 11,
// JSON-NEXT:                  "tokLen": 3
// JSON-NEXT:                 },
// JSON-NEXT:                 "end": {
// JSON-NEXT:                  "offset": {{.*}},
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
// JSON-NEXT:                  "offset": {{.*}},
// JSON-NEXT:                  "line": {{.*}},
// JSON-NEXT:                  "col": 12,
// JSON-NEXT:                  "tokLen": 1
// JSON-NEXT:                 },
// JSON-NEXT:                 "end": {
// JSON-NEXT:                  "offset": {{.*}},
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
// JSON-NEXT:     "offset": {{.*}},
// JSON-NEXT:     "line": {{.*}},
// JSON-NEXT:     "col": 6,
// JSON-NEXT:     "tokLen": 3
// JSON-NEXT:    },
// JSON-NEXT:    "range": {
// JSON-NEXT:     "begin": {
// JSON-NEXT:      "offset": {{.*}},
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 4
// JSON-NEXT:     },
// JSON-NEXT:     "end": {
// JSON-NEXT:      "offset": {{.*}},
// JSON-NEXT:      "line": {{.*}},
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
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "line": {{.*}},
// JSON-NEXT:        "col": 12,
// JSON-NEXT:        "tokLen": 1
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "line": {{.*}},
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
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "line": {{.*}},
// JSON-NEXT:          "col": 3,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 24,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "VarDecl",
// JSON-NEXT:          "loc": {
// JSON-NEXT:           "offset": {{.*}},
// JSON-NEXT:           "col": 7,
// JSON-NEXT:           "tokLen": 1
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 3,
// JSON-NEXT:            "tokLen": 3
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
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
// JSON-NEXT:              "offset": {{.*}},
// JSON-NEXT:              "col": 11,
// JSON-NEXT:              "tokLen": 3
// JSON-NEXT:             },
// JSON-NEXT:             "end": {
// JSON-NEXT:              "offset": {{.*}},
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
// JSON-NEXT:                "offset": {{.*}},
// JSON-NEXT:                "col": 11,
// JSON-NEXT:                "tokLen": 3
// JSON-NEXT:               },
// JSON-NEXT:               "end": {
// JSON-NEXT:                "offset": {{.*}},
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
// JSON-NEXT:                  "offset": {{.*}},
// JSON-NEXT:                  "col": 11,
// JSON-NEXT:                  "tokLen": 3
// JSON-NEXT:                 },
// JSON-NEXT:                 "end": {
// JSON-NEXT:                  "offset": {{.*}},
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
// JSON-NEXT:                },
// JSON-NEXT:                "qualType": {
// JSON-NEXT:                 "id": "0x{{.*}}",
// JSON-NEXT:                 "kind": "QualType",
// JSON-NEXT:                 "type": {
// JSON-NEXT:                  "qualType": "int ()"
// JSON-NEXT:                 },
// JSON-NEXT:                 "qualifiers": "",
// JSON-NEXT:                 "qualDetails": []
// JSON-NEXT:                }
// JSON-NEXT:               }
// JSON-NEXT:              ]
// JSON-NEXT:             }
// JSON-NEXT:            ]
// JSON-NEXT:           },
// JSON-NEXT:           {
// JSON-NEXT:            "qualDetails": [
// JSON-NEXT:             "signed",
// JSON-NEXT:             "integer"
// JSON-NEXT:            ],
// JSON-NEXT:            "typeDetails": [
// JSON-NEXT:             {
// JSON-NEXT:              "refId": "0x{{.*}}"
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
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "line": {{.*}},
// JSON-NEXT:          "col": 3,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 33,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "VarDecl",
// JSON-NEXT:          "loc": {
// JSON-NEXT:           "offset": {{.*}},
// JSON-NEXT:           "col": 7,
// JSON-NEXT:           "tokLen": 1
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 3,
// JSON-NEXT:            "tokLen": 3
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
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
// JSON-NEXT:              "offset": {{.*}},
// JSON-NEXT:              "col": 11,
// JSON-NEXT:              "tokLen": 3
// JSON-NEXT:             },
// JSON-NEXT:             "end": {
// JSON-NEXT:              "offset": {{.*}},
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
// JSON-NEXT:                "offset": {{.*}},
// JSON-NEXT:                "col": 11,
// JSON-NEXT:                "tokLen": 3
// JSON-NEXT:               },
// JSON-NEXT:               "end": {
// JSON-NEXT:                "offset": {{.*}},
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
// JSON-NEXT:                  "offset": {{.*}},
// JSON-NEXT:                  "col": 11,
// JSON-NEXT:                  "tokLen": 3
// JSON-NEXT:                 },
// JSON-NEXT:                 "end": {
// JSON-NEXT:                  "offset": {{.*}},
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
// JSON-NEXT:                    "offset": {{.*}},
// JSON-NEXT:                    "col": 11,
// JSON-NEXT:                    "tokLen": 3
// JSON-NEXT:                   },
// JSON-NEXT:                   "end": {
// JSON-NEXT:                    "offset": {{.*}},
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
// JSON-NEXT:                      "offset": {{.*}},
// JSON-NEXT:                      "col": 11,
// JSON-NEXT:                      "tokLen": 3
// JSON-NEXT:                     },
// JSON-NEXT:                     "end": {
// JSON-NEXT:                      "offset": {{.*}},
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
// JSON-NEXT:           },
// JSON-NEXT:           {
// JSON-NEXT:            "qualDetails": [
// JSON-NEXT:             "signed",
// JSON-NEXT:             "integer"
// JSON-NEXT:            ],
// JSON-NEXT:            "typeDetails": [
// JSON-NEXT:             {
// JSON-NEXT:              "refId": "0x{{.*}}"
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
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "line": {{.*}},
// JSON-NEXT:          "col": 3,
// JSON-NEXT:          "tokLen": 6
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 42,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "VarDecl",
// JSON-NEXT:          "loc": {
// JSON-NEXT:           "offset": {{.*}},
// JSON-NEXT:           "col": 10,
// JSON-NEXT:           "tokLen": 1
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 3,
// JSON-NEXT:            "tokLen": 6
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
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
// JSON-NEXT:              "offset": {{.*}},
// JSON-NEXT:              "col": 14,
// JSON-NEXT:              "tokLen": 3
// JSON-NEXT:             },
// JSON-NEXT:             "end": {
// JSON-NEXT:              "offset": {{.*}},
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
// JSON-NEXT:                "offset": {{.*}},
// JSON-NEXT:                "col": 14,
// JSON-NEXT:                "tokLen": 3
// JSON-NEXT:               },
// JSON-NEXT:               "end": {
// JSON-NEXT:                "offset": {{.*}},
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
// JSON-NEXT:                  "offset": {{.*}},
// JSON-NEXT:                  "col": 14,
// JSON-NEXT:                  "tokLen": 3
// JSON-NEXT:                 },
// JSON-NEXT:                 "end": {
// JSON-NEXT:                  "offset": {{.*}},
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
// JSON-NEXT:                    "offset": {{.*}},
// JSON-NEXT:                    "col": 14,
// JSON-NEXT:                    "tokLen": 3
// JSON-NEXT:                   },
// JSON-NEXT:                   "end": {
// JSON-NEXT:                    "offset": {{.*}},
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
// JSON-NEXT:                      "offset": {{.*}},
// JSON-NEXT:                      "col": 14,
// JSON-NEXT:                      "tokLen": 3
// JSON-NEXT:                     },
// JSON-NEXT:                     "end": {
// JSON-NEXT:                      "offset": {{.*}},
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
// JSON-NEXT:           },
// JSON-NEXT:           {
// JSON-NEXT:            "qualDetails": [
// JSON-NEXT:             "fpp"
// JSON-NEXT:            ],
// JSON-NEXT:            "typeDetails": [
// JSON-NEXT:             {
// JSON-NEXT:              "refId": "0x{{.*}}"
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
// JSON-NEXT:     "offset": {{.*}},
// JSON-NEXT:     "line": {{.*}},
// JSON-NEXT:     "col": 33,
// JSON-NEXT:     "tokLen": 1
// JSON-NEXT:    },
// JSON-NEXT:    "range": {
// JSON-NEXT:     "begin": {
// JSON-NEXT:      "offset": {{.*}},
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 8
// JSON-NEXT:     },
// JSON-NEXT:     "end": {
// JSON-NEXT:      "offset": {{.*}},
// JSON-NEXT:      "line": {{.*}},
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
// JSON-NEXT:       "offset": {{.*}},
// JSON-NEXT:       "line": {{.*}},
// JSON-NEXT:       "col": 23,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "col": 11,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": {{.*}},
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
// JSON-NEXT:       "offset": {{.*}},
// JSON-NEXT:       "col": 33,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "col": 26,
// JSON-NEXT:        "tokLen": 6
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "line": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "line": {{.*}},
// JSON-NEXT:         "col": 33,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 26,
// JSON-NEXT:          "tokLen": 6
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "line": {{.*}},
// JSON-NEXT:         "col": 31,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 3,
// JSON-NEXT:          "tokLen": 8
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:           "offset": {{.*}},
// JSON-NEXT:           "col": 18,
// JSON-NEXT:           "tokLen": 1
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 13,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
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
// JSON-NEXT:           "offset": {{.*}},
// JSON-NEXT:           "col": 31,
// JSON-NEXT:           "tokLen": 1
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 24,
// JSON-NEXT:            "tokLen": 6
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
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
// JSON-NEXT:             "offset": {{.*}},
// JSON-NEXT:             "col": 31,
// JSON-NEXT:             "tokLen": 1
// JSON-NEXT:            },
// JSON-NEXT:            "range": {
// JSON-NEXT:             "begin": {
// JSON-NEXT:              "offset": {{.*}},
// JSON-NEXT:              "col": 24,
// JSON-NEXT:              "tokLen": 6
// JSON-NEXT:             },
// JSON-NEXT:             "end": {
// JSON-NEXT:              "offset": {{.*}},
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
// JSON-NEXT:     "offset": {{.*}},
// JSON-NEXT:     "line": {{.*}},
// JSON-NEXT:     "col": 31,
// JSON-NEXT:     "tokLen": 1
// JSON-NEXT:    },
// JSON-NEXT:    "range": {
// JSON-NEXT:     "begin": {
// JSON-NEXT:      "offset": {{.*}},
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 8
// JSON-NEXT:     },
// JSON-NEXT:     "end": {
// JSON-NEXT:      "offset": {{.*}},
// JSON-NEXT:      "line": {{.*}},
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
// JSON-NEXT:       "offset": {{.*}},
// JSON-NEXT:       "line": {{.*}},
// JSON-NEXT:       "col": 23,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "col": 11,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": {{.*}},
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
// JSON-NEXT:       "offset": {{.*}},
// JSON-NEXT:       "col": 31,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "col": 26,
// JSON-NEXT:        "tokLen": 4
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "line": {{.*}},
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
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "line": {{.*}},
// JSON-NEXT:          "col": 35,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "line": {{.*}},
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
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "line": {{.*}},
// JSON-NEXT:            "col": 3,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 15,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "inner": [
// JSON-NEXT:           {
// JSON-NEXT:            "id": "0x{{.*}}",
// JSON-NEXT:            "kind": "VarDecl",
// JSON-NEXT:            "loc": {
// JSON-NEXT:             "offset": {{.*}},
// JSON-NEXT:             "col": 14,
// JSON-NEXT:             "tokLen": 1
// JSON-NEXT:            },
// JSON-NEXT:            "range": {
// JSON-NEXT:             "begin": {
// JSON-NEXT:              "offset": {{.*}},
// JSON-NEXT:              "col": 3,
// JSON-NEXT:              "tokLen": 1
// JSON-NEXT:             },
// JSON-NEXT:             "end": {
// JSON-NEXT:              "offset": {{.*}},
// JSON-NEXT:              "col": 14,
// JSON-NEXT:              "tokLen": 1
// JSON-NEXT:             }
// JSON-NEXT:            },
// JSON-NEXT:            "name": "a",
// JSON-NEXT:            "type": {
// JSON-NEXT:             "qualType": "A<T[3]...>"
// JSON-NEXT:            },
// JSON-NEXT:            "typeDetails": [
// JSON-NEXT:             {
// JSON-NEXT:              "id": "0x{{.*}}",
// JSON-NEXT:              "kind": "ElaboratedType",
// JSON-NEXT:              "type": {
// JSON-NEXT:               "qualType": "A<T[3]...>"
// JSON-NEXT:              },
// JSON-NEXT:              "isDependent": true,
// JSON-NEXT:              "isInstantiationDependent": true,
// JSON-NEXT:              "qualDetails": [],
// JSON-NEXT:              "typeDetails": [
// JSON-NEXT:               {
// JSON-NEXT:                "id": "0x{{.*}}",
// JSON-NEXT:                "kind": "TemplateSpecializationType",
// JSON-NEXT:                "type": {
// JSON-NEXT:                 "qualType": "A<T[3]...>"
// JSON-NEXT:                },
// JSON-NEXT:                "isDependent": true,
// JSON-NEXT:                "isInstantiationDependent": true,
// JSON-NEXT:                "templateName": "A",
// JSON-NEXT:                "qualDetails": [],
// JSON-NEXT:                "inner": [
// JSON-NEXT:                 {
// JSON-NEXT:                  "kind": "TemplateArgument",
// JSON-NEXT:                  "type": {
// JSON-NEXT:                   "qualType": "T[3]..."
// JSON-NEXT:                  },
// JSON-NEXT:                  "typeDetails": [
// JSON-NEXT:                   {
// JSON-NEXT:                    "id": "0x{{.*}}",
// JSON-NEXT:                    "kind": "PackExpansionType",
// JSON-NEXT:                    "type": {
// JSON-NEXT:                     "qualType": "T[3]..."
// JSON-NEXT:                    },
// JSON-NEXT:                    "isDependent": true,
// JSON-NEXT:                    "isInstantiationDependent": true,
// JSON-NEXT:                    "qualDetails": [],
// JSON-NEXT:                    "typeDetails": [
// JSON-NEXT:                     {
// JSON-NEXT:                      "id": "0x{{.*}}",
// JSON-NEXT:                      "kind": "ConstantArrayType",
// JSON-NEXT:                      "type": {
// JSON-NEXT:                       "qualType": "T[3]"
// JSON-NEXT:                      },
// JSON-NEXT:                      "isDependent": true,
// JSON-NEXT:                      "isInstantiationDependent": true,
// JSON-NEXT:                      "containsUnexpandedPack": true,
// JSON-NEXT:                      "size": 3,
// JSON-NEXT:                      "qualDetails": [
// JSON-NEXT:                       "array"
// JSON-NEXT:                      ],
// JSON-NEXT:                      "typeDetails": [
// JSON-NEXT:                       {
// JSON-NEXT:                        "id": "0x{{.*}}",
// JSON-NEXT:                        "kind": "TemplateTypeParmType",
// JSON-NEXT:                        "type": {
// JSON-NEXT:                         "qualType": "T"
// JSON-NEXT:                        },
// JSON-NEXT:                        "isDependent": true,
// JSON-NEXT:                        "isInstantiationDependent": true,
// JSON-NEXT:                        "containsUnexpandedPack": true,
// JSON-NEXT:                        "depth": 0,
// JSON-NEXT:                        "index": 0,
// JSON-NEXT:                        "isPack": true,
// JSON-NEXT:                        "decl": {
// JSON-NEXT:                         "id": "0x{{.*}}",
// JSON-NEXT:                         "kind": "TemplateTypeParmDecl",
// JSON-NEXT:                         "name": "T"
// JSON-NEXT:                        },
// JSON-NEXT:                        "qualDetails": []
// JSON-NEXT:                       }
// JSON-NEXT:                      ]
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
// JSON-NEXT:       }
// JSON-NEXT:      ]
// JSON-NEXT:     }
// JSON-NEXT:    ]
// JSON-NEXT:   },
// JSON-NEXT:   {
// JSON-NEXT:    "id": "0x{{.*}}",
// JSON-NEXT:    "kind": "NamespaceDecl",
// JSON-NEXT:    "loc": {
// JSON-NEXT:     "offset": {{.*}},
// JSON-NEXT:     "line": {{.*}},
// JSON-NEXT:     "col": 11,
// JSON-NEXT:     "tokLen": 5
// JSON-NEXT:    },
// JSON-NEXT:    "range": {
// JSON-NEXT:     "begin": {
// JSON-NEXT:      "offset": {{.*}},
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 9
// JSON-NEXT:     },
// JSON-NEXT:     "end": {
// JSON-NEXT:      "offset": {{.*}},
// JSON-NEXT:      "line": {{.*}},
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
// JSON-NEXT:       "offset": {{.*}},
// JSON-NEXT:       "line": {{.*}},
// JSON-NEXT:       "col": 6,
// JSON-NEXT:       "tokLen": 4
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 4
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "col": 14,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 11,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 11,
// JSON-NEXT:          "tokLen": 3
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "int"
// JSON-NEXT:        },
// JSON-NEXT:        "typeDetails": [
// JSON-NEXT:         {
// JSON-NEXT:          "qualDetails": [
// JSON-NEXT:           "signed",
// JSON-NEXT:           "integer"
// JSON-NEXT:          ],
// JSON-NEXT:          "typeDetails": [
// JSON-NEXT:           {
// JSON-NEXT:            "refId": "0x{{.*}}"
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
// JSON-NEXT:       "offset": {{.*}},
// JSON-NEXT:       "line": {{.*}},
// JSON-NEXT:       "col": 6,
// JSON-NEXT:       "tokLen": 4
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 4
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "col": 16,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 11,
// JSON-NEXT:          "tokLen": 5
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 11,
// JSON-NEXT:          "tokLen": 5
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "float"
// JSON-NEXT:        },
// JSON-NEXT:        "typeDetails": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "BuiltinType",
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "float"
// JSON-NEXT:          },
// JSON-NEXT:          "qualDetails": [
// JSON-NEXT:           "fpp"
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
// JSON-NEXT:       "offset": {{.*}},
// JSON-NEXT:       "line": {{.*}},
// JSON-NEXT:       "col": 6,
// JSON-NEXT:       "tokLen": 4
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "line": {{.*}},
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "line": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "line": {{.*}},
// JSON-NEXT:         "col": 19,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 10,
// JSON-NEXT:          "tokLen": 8
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "line": {{.*}},
// JSON-NEXT:         "col": 6,
// JSON-NEXT:         "tokLen": 4
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 1,
// JSON-NEXT:          "tokLen": 4
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "line": {{.*}},
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
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "line": {{.*}},
// JSON-NEXT:            "col": 13,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "line": {{.*}},
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
// JSON-NEXT:              "offset": {{.*}},
// JSON-NEXT:              "line": {{.*}},
// JSON-NEXT:              "col": 3,
// JSON-NEXT:              "tokLen": 4
// JSON-NEXT:             },
// JSON-NEXT:             "end": {
// JSON-NEXT:              "offset": {{.*}},
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
// JSON-NEXT:                "offset": {{.*}},
// JSON-NEXT:                "col": 3,
// JSON-NEXT:                "tokLen": 4
// JSON-NEXT:               },
// JSON-NEXT:               "end": {
// JSON-NEXT:                "offset": {{.*}},
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
// JSON-NEXT:                "offset": {{.*}},
// JSON-NEXT:                "col": 8,
// JSON-NEXT:                "tokLen": 1
// JSON-NEXT:               },
// JSON-NEXT:               "end": {
// JSON-NEXT:                "offset": {{.*}},
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
// JSON-NEXT:     "offset": {{.*}},
// JSON-NEXT:     "line": {{.*}},
// JSON-NEXT:     "col": 11,
// JSON-NEXT:     "tokLen": 5
// JSON-NEXT:    },
// JSON-NEXT:    "range": {
// JSON-NEXT:     "begin": {
// JSON-NEXT:      "offset": {{.*}},
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 9
// JSON-NEXT:     },
// JSON-NEXT:     "end": {
// JSON-NEXT:      "offset": {{.*}},
// JSON-NEXT:      "line": {{.*}},
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
// JSON-NEXT:       "offset": {{.*}},
// JSON-NEXT:       "line": {{.*}},
// JSON-NEXT:       "col": 31,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "col": 3,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "col": 21,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 12,
// JSON-NEXT:          "tokLen": 8
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "col": 31,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 24,
// JSON-NEXT:          "tokLen": 6
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:           "offset": {{.*}},
// JSON-NEXT:           "col": 31,
// JSON-NEXT:           "tokLen": 1
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 24,
// JSON-NEXT:            "tokLen": 6
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "col": 31,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 3,
// JSON-NEXT:          "tokLen": 8
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:          "typeDetails": [
// JSON-NEXT:           {
// JSON-NEXT:            "qualDetails": [
// JSON-NEXT:             "signed",
// JSON-NEXT:             "integer"
// JSON-NEXT:            ],
// JSON-NEXT:            "typeDetails": [
// JSON-NEXT:             {
// JSON-NEXT:              "refId": "0x{{.*}}"
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
// JSON-NEXT:      "kind": "FunctionTemplateDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": {{.*}},
// JSON-NEXT:       "col": 31,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "col": 3,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "col": 21,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 12,
// JSON-NEXT:          "tokLen": 8
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "col": 31,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 31,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 31,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "isImplicit": true,
// JSON-NEXT:        "name": "<deduction guide for A>",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "auto () -> A<T>"
// JSON-NEXT:        }
// JSON-NEXT:       }
// JSON-NEXT:      ]
// JSON-NEXT:     },
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "FunctionTemplateDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": {{.*}},
// JSON-NEXT:       "col": 31,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "col": 3,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "col": 21,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 12,
// JSON-NEXT:          "tokLen": 8
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "col": 31,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 31,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 31,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         }
// JSON-NEXT:        },
// JSON-NEXT:        "isImplicit": true,
// JSON-NEXT:        "name": "<deduction guide for A>",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "auto (A<T>) -> A<T>"
// JSON-NEXT:        },
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "ParmVarDecl",
// JSON-NEXT:          "loc": {
// JSON-NEXT:           "offset": {{.*}},
// JSON-NEXT:           "col": 31,
// JSON-NEXT:           "tokLen": 1
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 31,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 31,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "A<T>"
// JSON-NEXT:          },
// JSON-NEXT:          "typeDetails": [
// JSON-NEXT:           {
// JSON-NEXT:            "id": "0x{{.*}}",
// JSON-NEXT:            "kind": "InjectedClassNameType",
// JSON-NEXT:            "type": {
// JSON-NEXT:             "qualType": "A<T>"
// JSON-NEXT:            },
// JSON-NEXT:            "isDependent": true,
// JSON-NEXT:            "isInstantiationDependent": true,
// JSON-NEXT:            "decl": {
// JSON-NEXT:             "id": "0x{{.*}}",
// JSON-NEXT:             "kind": "CXXRecordDecl",
// JSON-NEXT:             "name": "A"
// JSON-NEXT:            },
// JSON-NEXT:            "qualDetails": []
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
// JSON-NEXT:       "offset": {{.*}},
// JSON-NEXT:       "line": {{.*}},
// JSON-NEXT:       "col": 24,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "col": 3,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "col": 21,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 12,
// JSON-NEXT:          "tokLen": 8
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "col": 24,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 24,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:           "offset": {{.*}},
// JSON-NEXT:           "col": 27,
// JSON-NEXT:           "tokLen": 1
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 26,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 26,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "T"
// JSON-NEXT:          },
// JSON-NEXT:          "typeDetails": [
// JSON-NEXT:           {
// JSON-NEXT:            "id": "0x{{.*}}",
// JSON-NEXT:            "kind": "TemplateTypeParmType",
// JSON-NEXT:            "type": {
// JSON-NEXT:             "qualType": "T"
// JSON-NEXT:            },
// JSON-NEXT:            "isDependent": true,
// JSON-NEXT:            "isInstantiationDependent": true,
// JSON-NEXT:            "depth": 0,
// JSON-NEXT:            "index": 0,
// JSON-NEXT:            "decl": {
// JSON-NEXT:             "id": "0x{{.*}}",
// JSON-NEXT:             "kind": "TemplateTypeParmDecl",
// JSON-NEXT:             "name": "T"
// JSON-NEXT:            },
// JSON-NEXT:            "qualDetails": []
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
// JSON-NEXT:     "offset": {{.*}},
// JSON-NEXT:     "line": {{.*}},
// JSON-NEXT:     "col": 11,
// JSON-NEXT:     "tokLen": 5
// JSON-NEXT:    },
// JSON-NEXT:    "range": {
// JSON-NEXT:     "begin": {
// JSON-NEXT:      "offset": {{.*}},
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 9
// JSON-NEXT:     },
// JSON-NEXT:     "end": {
// JSON-NEXT:      "offset": {{.*}},
// JSON-NEXT:      "line": {{.*}},
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
// JSON-NEXT:       "offset": {{.*}},
// JSON-NEXT:       "line": {{.*}},
// JSON-NEXT:       "col": 8,
// JSON-NEXT:       "tokLen": 3
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "line": {{.*}},
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "line": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "line": {{.*}},
// JSON-NEXT:         "col": 20,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 11,
// JSON-NEXT:          "tokLen": 8
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "col": 28,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 23,
// JSON-NEXT:          "tokLen": 4
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "line": {{.*}},
// JSON-NEXT:         "col": 8,
// JSON-NEXT:         "tokLen": 3
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 1,
// JSON-NEXT:          "tokLen": 6
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "line": {{.*}},
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
// JSON-NEXT:           "offset": {{.*}},
// JSON-NEXT:           "line": {{.*}},
// JSON-NEXT:           "col": 8,
// JSON-NEXT:           "tokLen": 3
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 1,
// JSON-NEXT:            "tokLen": 6
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
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
// JSON-NEXT:           "offset": {{.*}},
// JSON-NEXT:           "line": {{.*}},
// JSON-NEXT:           "col": 15,
// JSON-NEXT:           "tokLen": 2
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 3,
// JSON-NEXT:            "tokLen": 6
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "line": {{.*}},
// JSON-NEXT:         "col": 8,
// JSON-NEXT:         "tokLen": 3
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "line": {{.*}},
// JSON-NEXT:          "col": 1,
// JSON-NEXT:          "tokLen": 8
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "line": {{.*}},
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
// JSON-NEXT:           "offset": {{.*}},
// JSON-NEXT:           "line": {{.*}},
// JSON-NEXT:           "col": 8,
// JSON-NEXT:           "tokLen": 3
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 1,
// JSON-NEXT:            "tokLen": 6
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
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
// JSON-NEXT:           "offset": {{.*}},
// JSON-NEXT:           "line": {{.*}},
// JSON-NEXT:           "col": 15,
// JSON-NEXT:           "tokLen": 2
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 3,
// JSON-NEXT:            "tokLen": 6
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
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
// JSON-NEXT:       "offset": {{.*}},
// JSON-NEXT:       "line": {{.*}},
// JSON-NEXT:       "col": 6,
// JSON-NEXT:       "tokLen": 4
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 4
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "line": {{.*}},
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
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "line": {{.*}},
// JSON-NEXT:          "col": 13,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "line": {{.*}},
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
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "line": {{.*}},
// JSON-NEXT:            "col": 3,
// JSON-NEXT:            "tokLen": 3
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
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
// JSON-NEXT:              "offset": {{.*}},
// JSON-NEXT:              "col": 3,
// JSON-NEXT:              "tokLen": 3
// JSON-NEXT:             },
// JSON-NEXT:             "end": {
// JSON-NEXT:              "offset": {{.*}},
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
// JSON-NEXT:                "offset": {{.*}},
// JSON-NEXT:                "col": 3,
// JSON-NEXT:                "tokLen": 3
// JSON-NEXT:               },
// JSON-NEXT:               "end": {
// JSON-NEXT:                "offset": {{.*}},
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
// JSON-NEXT:              },
// JSON-NEXT:              "qualType": {
// JSON-NEXT:               "id": "0x{{.*}}",
// JSON-NEXT:               "kind": "QualType",
// JSON-NEXT:               "type": {
// JSON-NEXT:                "qualType": "void ()"
// JSON-NEXT:               },
// JSON-NEXT:               "qualifiers": "",
// JSON-NEXT:               "qualDetails": []
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
// JSON-NEXT:       "offset": {{.*}},
// JSON-NEXT:       "line": {{.*}},
// JSON-NEXT:       "col": 17,
// JSON-NEXT:       "tokLen": 3
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "line": {{.*}},
// JSON-NEXT:         "col": 8,
// JSON-NEXT:         "tokLen": 3
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 1,
// JSON-NEXT:          "tokLen": 6
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "line": {{.*}},
// JSON-NEXT:         "col": 15,
// JSON-NEXT:         "tokLen": 2
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 3,
// JSON-NEXT:          "tokLen": 6
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:     "offset": {{.*}},
// JSON-NEXT:     "line": {{.*}},
// JSON-NEXT:     "col": 11,
// JSON-NEXT:     "tokLen": 5
// JSON-NEXT:    },
// JSON-NEXT:    "range": {
// JSON-NEXT:     "begin": {
// JSON-NEXT:      "offset": {{.*}},
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 9
// JSON-NEXT:     },
// JSON-NEXT:     "end": {
// JSON-NEXT:      "offset": {{.*}},
// JSON-NEXT:      "line": {{.*}},
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
// JSON-NEXT:       "offset": {{.*}},
// JSON-NEXT:       "line": {{.*}},
// JSON-NEXT:       "col": 21,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "col": 14,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 10,
// JSON-NEXT:          "tokLen": 4
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "col": 21,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 16,
// JSON-NEXT:          "tokLen": 4
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 25,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "col": 21,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 16,
// JSON-NEXT:          "tokLen": 4
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 25,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
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
// JSON-NEXT:       "offset": {{.*}},
// JSON-NEXT:       "line": {{.*}},
// JSON-NEXT:       "col": 8,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 4
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": {{.*}},
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
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 15,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 15,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
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
// JSON-NEXT:          },
// JSON-NEXT:          "qualType": {
// JSON-NEXT:           "refId": "0x{{.*}}",
// JSON-NEXT:           "qualDetails": []
// JSON-NEXT:          }
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "PointerType",
// JSON-NEXT:        "type": {
// JSON-NEXT:         "qualType": "void (*)()"
// JSON-NEXT:        },
// JSON-NEXT:        "qualDetails": [
// JSON-NEXT:         "ptr",
// JSON-NEXT:         "func_ptr"
// JSON-NEXT:        ],
// JSON-NEXT:        "typeDetails": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "ParenType",
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "void ()"
// JSON-NEXT:          },
// JSON-NEXT:          "qualDetails": [],
// JSON-NEXT:          "typeDetails": [
// JSON-NEXT:           {
// JSON-NEXT:            "cc": "cdecl",
// JSON-NEXT:            "qualDetails": [],
// JSON-NEXT:            "typeDetails": [
// JSON-NEXT:             {
// JSON-NEXT:              "refId": "0x{{.*}}"
// JSON-NEXT:             },
// JSON-NEXT:             {
// JSON-NEXT:              "id": "0x{{.*}}",
// JSON-NEXT:              "kind": "BuiltinType",
// JSON-NEXT:              "type": {
// JSON-NEXT:               "qualType": "void"
// JSON-NEXT:              },
// JSON-NEXT:              "qualDetails": [
// JSON-NEXT:               "void"
// JSON-NEXT:              ]
// JSON-NEXT:             },
// JSON-NEXT:             {
// JSON-NEXT:              "cc": "cdecl",
// JSON-NEXT:              "returnTypeDetail": {
// JSON-NEXT:               "qualType": {
// JSON-NEXT:                "refId": "0x{{.*}}",
// JSON-NEXT:                "qualDetails": [
// JSON-NEXT:                 "void"
// JSON-NEXT:                ]
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
// JSON-NEXT:      "kind": "FunctionTemplateDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": {{.*}},
// JSON-NEXT:       "line": {{.*}},
// JSON-NEXT:       "col": 29,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "col": 19,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 10,
// JSON-NEXT:          "tokLen": 8
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 21,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
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
// JSON-NEXT:              "offset": {{.*}},
// JSON-NEXT:              "col": 21,
// JSON-NEXT:              "tokLen": 1
// JSON-NEXT:             },
// JSON-NEXT:             "end": {
// JSON-NEXT:              "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "col": 29,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 24,
// JSON-NEXT:          "tokLen": 4
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 33,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "col": 29,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 24,
// JSON-NEXT:          "tokLen": 4
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 33,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
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
// JSON-NEXT:       "offset": {{.*}},
// JSON-NEXT:       "line": {{.*}},
// JSON-NEXT:       "col": 8,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 4
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": {{.*}},
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
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 15,
// JSON-NEXT:          "tokLen": 1
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 15,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
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
// JSON-NEXT:          },
// JSON-NEXT:          "qualType": {
// JSON-NEXT:           "refId": "0x{{.*}}",
// JSON-NEXT:           "qualDetails": []
// JSON-NEXT:          }
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "qualDetails": [
// JSON-NEXT:         "ptr",
// JSON-NEXT:         "func_ptr"
// JSON-NEXT:        ],
// JSON-NEXT:        "typeDetails": [
// JSON-NEXT:         {
// JSON-NEXT:          "refId": "0x{{.*}}"
// JSON-NEXT:         },
// JSON-NEXT:         {
// JSON-NEXT:          "qualDetails": [],
// JSON-NEXT:          "typeDetails": [
// JSON-NEXT:           {
// JSON-NEXT:            "refId": "0x{{.*}}"
// JSON-NEXT:           },
// JSON-NEXT:           {
// JSON-NEXT:            "cc": "cdecl",
// JSON-NEXT:            "qualDetails": [],
// JSON-NEXT:            "typeDetails": [
// JSON-NEXT:             {
// JSON-NEXT:              "refId": "0x{{.*}}"
// JSON-NEXT:             },
// JSON-NEXT:             {
// JSON-NEXT:              "qualDetails": [
// JSON-NEXT:               "void"
// JSON-NEXT:              ],
// JSON-NEXT:              "typeDetails": [
// JSON-NEXT:               {
// JSON-NEXT:                "refId": "0x{{.*}}"
// JSON-NEXT:               }
// JSON-NEXT:              ]
// JSON-NEXT:             },
// JSON-NEXT:             {
// JSON-NEXT:              "cc": "cdecl",
// JSON-NEXT:              "returnTypeDetail": {
// JSON-NEXT:               "qualType": {
// JSON-NEXT:                "refId": "0x{{.*}}",
// JSON-NEXT:                "qualDetails": [
// JSON-NEXT:                 "void"
// JSON-NEXT:                ]
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
// JSON-NEXT:     }
// JSON-NEXT:    ]
// JSON-NEXT:   },
// JSON-NEXT:   {
// JSON-NEXT:    "id": "0x{{.*}}",
// JSON-NEXT:    "kind": "NamespaceDecl",
// JSON-NEXT:    "loc": {
// JSON-NEXT:     "offset": {{.*}},
// JSON-NEXT:     "line": {{.*}},
// JSON-NEXT:     "col": 11,
// JSON-NEXT:     "tokLen": 5
// JSON-NEXT:    },
// JSON-NEXT:    "range": {
// JSON-NEXT:     "begin": {
// JSON-NEXT:      "offset": {{.*}},
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 9
// JSON-NEXT:     },
// JSON-NEXT:     "end": {
// JSON-NEXT:      "offset": {{.*}},
// JSON-NEXT:      "line": {{.*}},
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
// JSON-NEXT:       "offset": {{.*}},
// JSON-NEXT:       "line": {{.*}},
// JSON-NEXT:       "col": 16,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "line": {{.*}},
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "line": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "line": {{.*}},
// JSON-NEXT:         "col": 17,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 11,
// JSON-NEXT:          "tokLen": 5
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "line": {{.*}},
// JSON-NEXT:         "col": 16,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 1,
// JSON-NEXT:          "tokLen": 9
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 20,
// JSON-NEXT:            "tokLen": 4
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 20,
// JSON-NEXT:            "tokLen": 4
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "bool"
// JSON-NEXT:          },
// JSON-NEXT:          "valueCategory": "prvalue",
// JSON-NEXT:          "value": true
// JSON-NEXT:         },
// JSON-NEXT:         {
// JSON-NEXT:          "qualType": {
// JSON-NEXT:           "id": "0x{{.*}}",
// JSON-NEXT:           "kind": "QualType",
// JSON-NEXT:           "type": {
// JSON-NEXT:            "qualType": "const bool"
// JSON-NEXT:           },
// JSON-NEXT:           "qualifiers": "const",
// JSON-NEXT:           "qualDetails": [
// JSON-NEXT:            "unsigned",
// JSON-NEXT:            "integer"
// JSON-NEXT:           ]
// JSON-NEXT:          },
// JSON-NEXT:          "typeDetails": [
// JSON-NEXT:           {
// JSON-NEXT:            "id": "0x{{.*}}",
// JSON-NEXT:            "kind": "BuiltinType",
// JSON-NEXT:            "type": {
// JSON-NEXT:             "qualType": "bool"
// JSON-NEXT:            },
// JSON-NEXT:            "qualDetails": [
// JSON-NEXT:             "unsigned",
// JSON-NEXT:             "integer"
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
// JSON-NEXT:      "kind": "FunctionTemplateDecl",
// JSON-NEXT:      "loc": {
// JSON-NEXT:       "offset": {{.*}},
// JSON-NEXT:       "line": {{.*}},
// JSON-NEXT:       "col": 6,
// JSON-NEXT:       "tokLen": 4
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "line": {{.*}},
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "line": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "line": {{.*}},
// JSON-NEXT:         "col": 17,
// JSON-NEXT:         "tokLen": 3
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 11,
// JSON-NEXT:          "tokLen": 5
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "line": {{.*}},
// JSON-NEXT:         "col": 6,
// JSON-NEXT:         "tokLen": 4
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 1,
// JSON-NEXT:          "tokLen": 4
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "line": {{.*}},
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
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "line": {{.*}},
// JSON-NEXT:            "col": 13,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "line": {{.*}},
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
// JSON-NEXT:              "offset": {{.*}},
// JSON-NEXT:              "line": {{.*}},
// JSON-NEXT:              "col": 3,
// JSON-NEXT:              "tokLen": 1
// JSON-NEXT:             },
// JSON-NEXT:             "end": {
// JSON-NEXT:              "offset": {{.*}},
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
// JSON-NEXT:              "typeDetails": [
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
// JSON-NEXT:                },
// JSON-NEXT:                "qualDetails": []
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
// JSON-NEXT:     "offset": {{.*}},
// JSON-NEXT:     "line": {{.*}},
// JSON-NEXT:     "col": 11,
// JSON-NEXT:     "tokLen": 5
// JSON-NEXT:    },
// JSON-NEXT:    "range": {
// JSON-NEXT:     "begin": {
// JSON-NEXT:      "offset": {{.*}},
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 9
// JSON-NEXT:     },
// JSON-NEXT:     "end": {
// JSON-NEXT:      "offset": {{.*}},
// JSON-NEXT:      "line": {{.*}},
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
// JSON-NEXT:       "offset": {{.*}},
// JSON-NEXT:       "line": {{.*}},
// JSON-NEXT:       "col": 46,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "col": 3,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "col": 35,
// JSON-NEXT:         "tokLen": 2
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 13,
// JSON-NEXT:          "tokLen": 8
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:           "offset": {{.*}},
// JSON-NEXT:           "col": 27,
// JSON-NEXT:           "tokLen": 1
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 22,
// JSON-NEXT:            "tokLen": 5
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "col": 46,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 39,
// JSON-NEXT:          "tokLen": 6
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:           "offset": {{.*}},
// JSON-NEXT:           "col": 46,
// JSON-NEXT:           "tokLen": 1
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 39,
// JSON-NEXT:            "tokLen": 6
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
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
// JSON-NEXT:       "offset": {{.*}},
// JSON-NEXT:       "line": {{.*}},
// JSON-NEXT:       "col": 29,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "col": 3,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "col": 21,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 13,
// JSON-NEXT:          "tokLen": 5
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "col": 29,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 23,
// JSON-NEXT:          "tokLen": 5
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:           "offset": {{.*}},
// JSON-NEXT:           "col": 29,
// JSON-NEXT:           "tokLen": 1
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 23,
// JSON-NEXT:            "tokLen": 5
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
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
// JSON-NEXT:       "offset": {{.*}},
// JSON-NEXT:       "line": {{.*}},
// JSON-NEXT:       "col": 19,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "col": 3,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "line": {{.*}},
// JSON-NEXT:         "col": 46,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 39,
// JSON-NEXT:          "tokLen": 6
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:     "offset": {{.*}},
// JSON-NEXT:     "line": {{.*}},
// JSON-NEXT:     "col": 11,
// JSON-NEXT:     "tokLen": 5
// JSON-NEXT:    },
// JSON-NEXT:    "range": {
// JSON-NEXT:     "begin": {
// JSON-NEXT:      "offset": {{.*}},
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 9
// JSON-NEXT:     },
// JSON-NEXT:     "end": {
// JSON-NEXT:      "offset": {{.*}},
// JSON-NEXT:      "line": {{.*}},
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
// JSON-NEXT:       "offset": {{.*}},
// JSON-NEXT:       "line": {{.*}},
// JSON-NEXT:       "col": 8,
// JSON-NEXT:       "tokLen": 8
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "line": {{.*}},
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "line": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "line": {{.*}},
// JSON-NEXT:         "col": 23,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 10,
// JSON-NEXT:          "tokLen": 8
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "line": {{.*}},
// JSON-NEXT:         "col": 8,
// JSON-NEXT:         "tokLen": 8
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 1,
// JSON-NEXT:          "tokLen": 6
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:       "offset": {{.*}},
// JSON-NEXT:       "line": {{.*}},
// JSON-NEXT:       "col": 8,
// JSON-NEXT:       "tokLen": 8
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "line": {{.*}},
// JSON-NEXT:        "col": 1,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "line": {{.*}},
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
// JSON-NEXT:     "offset": {{.*}},
// JSON-NEXT:     "line": {{.*}},
// JSON-NEXT:     "col": 11,
// JSON-NEXT:     "tokLen": 28
// JSON-NEXT:    },
// JSON-NEXT:    "range": {
// JSON-NEXT:     "begin": {
// JSON-NEXT:      "offset": {{.*}},
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 9
// JSON-NEXT:     },
// JSON-NEXT:     "end": {
// JSON-NEXT:      "offset": {{.*}},
// JSON-NEXT:      "line": {{.*}},
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
// JSON-NEXT:       "offset": {{.*}},
// JSON-NEXT:       "line": {{.*}},
// JSON-NEXT:       "col": 27,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "col": 3,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "col": 18,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 13,
// JSON-NEXT:          "tokLen": 5
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "col": 27,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 20,
// JSON-NEXT:          "tokLen": 6
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:       "offset": {{.*}},
// JSON-NEXT:       "line": {{.*}},
// JSON-NEXT:       "col": 40,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "col": 3,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": {{.*}},
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
// JSON-NEXT:        "typeDetails": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "MemberPointerType",
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "type-parameter-0-0 type-parameter-0-1::*"
// JSON-NEXT:          },
// JSON-NEXT:          "isDependent": true,
// JSON-NEXT:          "isInstantiationDependent": true,
// JSON-NEXT:          "isData": true,
// JSON-NEXT:          "qualDetails": [],
// JSON-NEXT:          "typeDetails": [
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
// JSON-NEXT:             "id": "{{.*}}"
// JSON-NEXT:            },
// JSON-NEXT:            "qualDetails": []
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
// JSON-NEXT:             "id": "{{.*}}"
// JSON-NEXT:            },
// JSON-NEXT:            "qualDetails": []
// JSON-NEXT:           }
// JSON-NEXT:          ]
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "TemplateTypeParmDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "col": 19,
// JSON-NEXT:         "tokLen": 2
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 13,
// JSON-NEXT:          "tokLen": 5
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "col": 29,
// JSON-NEXT:         "tokLen": 2
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 23,
// JSON-NEXT:          "tokLen": 5
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:     "offset": {{.*}},
// JSON-NEXT:     "line": {{.*}},
// JSON-NEXT:     "col": 11,
// JSON-NEXT:     "tokLen": 26
// JSON-NEXT:    },
// JSON-NEXT:    "range": {
// JSON-NEXT:     "begin": {
// JSON-NEXT:      "offset": {{.*}},
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 9
// JSON-NEXT:     },
// JSON-NEXT:     "end": {
// JSON-NEXT:      "offset": {{.*}},
// JSON-NEXT:      "line": {{.*}},
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
// JSON-NEXT:       "offset": {{.*}},
// JSON-NEXT:       "line": {{.*}},
// JSON-NEXT:       "col": 29,
// JSON-NEXT:       "tokLen": 1
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "col": 3,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "line": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "line": {{.*}},
// JSON-NEXT:         "col": 19,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 13,
// JSON-NEXT:          "tokLen": 5
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "col": 29,
// JSON-NEXT:         "tokLen": 1
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 22,
// JSON-NEXT:          "tokLen": 6
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "line": {{.*}},
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
// JSON-NEXT:           "offset": {{.*}},
// JSON-NEXT:           "line": {{.*}},
// JSON-NEXT:           "col": 29,
// JSON-NEXT:           "tokLen": 1
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 22,
// JSON-NEXT:            "tokLen": 6
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
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
// JSON-NEXT:           "offset": {{.*}},
// JSON-NEXT:           "line": {{.*}},
// JSON-NEXT:           "col": 11,
// JSON-NEXT:           "tokLen": 1
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 5,
// JSON-NEXT:            "tokLen": 5
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 22,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "name": "X",
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "int U::*"
// JSON-NEXT:          },
// JSON-NEXT:          "typeDetails": [
// JSON-NEXT:           {
// JSON-NEXT:            "id": "0x{{.*}}",
// JSON-NEXT:            "kind": "MemberPointerType",
// JSON-NEXT:            "type": {
// JSON-NEXT:             "qualType": "int U::*"
// JSON-NEXT:            },
// JSON-NEXT:            "isDependent": true,
// JSON-NEXT:            "isInstantiationDependent": true,
// JSON-NEXT:            "isData": true,
// JSON-NEXT:            "qualDetails": [],
// JSON-NEXT:            "typeDetails": [
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
// JSON-NEXT:              },
// JSON-NEXT:              "qualDetails": []
// JSON-NEXT:             },
// JSON-NEXT:             {
// JSON-NEXT:              "qualDetails": [
// JSON-NEXT:               "signed",
// JSON-NEXT:               "integer"
// JSON-NEXT:              ],
// JSON-NEXT:              "typeDetails": [
// JSON-NEXT:               {
// JSON-NEXT:                "refId": "0x{{.*}}"
// JSON-NEXT:               }
// JSON-NEXT:              ]
// JSON-NEXT:             }
// JSON-NEXT:            ]
// JSON-NEXT:           }
// JSON-NEXT:          ]
// JSON-NEXT:         },
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "TypeAliasDecl",
// JSON-NEXT:          "loc": {
// JSON-NEXT:           "offset": {{.*}},
// JSON-NEXT:           "line": {{.*}},
// JSON-NEXT:           "col": 11,
// JSON-NEXT:           "tokLen": 1
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 5,
// JSON-NEXT:            "tokLen": 5
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 28,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "name": "Y",
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "int U::test::*"
// JSON-NEXT:          },
// JSON-NEXT:          "typeDetails": [
// JSON-NEXT:           {
// JSON-NEXT:            "id": "0x{{.*}}",
// JSON-NEXT:            "kind": "MemberPointerType",
// JSON-NEXT:            "type": {
// JSON-NEXT:             "qualType": "int U::test::*"
// JSON-NEXT:            },
// JSON-NEXT:            "isDependent": true,
// JSON-NEXT:            "isInstantiationDependent": true,
// JSON-NEXT:            "isData": true,
// JSON-NEXT:            "qualDetails": [],
// JSON-NEXT:            "typeDetails": [
// JSON-NEXT:             {
// JSON-NEXT:              "qualDetails": [
// JSON-NEXT:               "signed",
// JSON-NEXT:               "integer"
// JSON-NEXT:              ],
// JSON-NEXT:              "typeDetails": [
// JSON-NEXT:               {
// JSON-NEXT:                "refId": "0x{{.*}}"
// JSON-NEXT:               }
// JSON-NEXT:              ]
// JSON-NEXT:             }
// JSON-NEXT:            ]
// JSON-NEXT:           }
// JSON-NEXT:          ]
// JSON-NEXT:         },
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "TypeAliasDecl",
// JSON-NEXT:          "loc": {
// JSON-NEXT:           "offset": {{.*}},
// JSON-NEXT:           "line": {{.*}},
// JSON-NEXT:           "col": 11,
// JSON-NEXT:           "tokLen": 1
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 5,
// JSON-NEXT:            "tokLen": 5
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 39,
// JSON-NEXT:            "tokLen": 1
// JSON-NEXT:           }
// JSON-NEXT:          },
// JSON-NEXT:          "name": "Z",
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "int U::template V<int>::*"
// JSON-NEXT:          },
// JSON-NEXT:          "typeDetails": [
// JSON-NEXT:           {
// JSON-NEXT:            "id": "0x{{.*}}",
// JSON-NEXT:            "kind": "MemberPointerType",
// JSON-NEXT:            "type": {
// JSON-NEXT:             "qualType": "int U::template V<int>::*"
// JSON-NEXT:            },
// JSON-NEXT:            "isDependent": true,
// JSON-NEXT:            "isInstantiationDependent": true,
// JSON-NEXT:            "isData": true,
// JSON-NEXT:            "qualDetails": [],
// JSON-NEXT:            "typeDetails": [
// JSON-NEXT:             {
// JSON-NEXT:              "id": "0x{{.*}}",
// JSON-NEXT:              "kind": "DependentTemplateSpecializationType",
// JSON-NEXT:              "type": {
// JSON-NEXT:               "qualType": "template V<int>"
// JSON-NEXT:              },
// JSON-NEXT:              "isDependent": true,
// JSON-NEXT:              "isInstantiationDependent": true,
// JSON-NEXT:              "qualDetails": []
// JSON-NEXT:             },
// JSON-NEXT:             {
// JSON-NEXT:              "qualDetails": [
// JSON-NEXT:               "signed",
// JSON-NEXT:               "integer"
// JSON-NEXT:              ],
// JSON-NEXT:              "typeDetails": [
// JSON-NEXT:               {
// JSON-NEXT:                "refId": "0x{{.*}}"
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
// JSON-NEXT:     "offset": {{.*}},
// JSON-NEXT:     "line": {{.*}},
// JSON-NEXT:     "col": 11,
// JSON-NEXT:     "tokLen": 19
// JSON-NEXT:    },
// JSON-NEXT:    "range": {
// JSON-NEXT:     "begin": {
// JSON-NEXT:      "offset": {{.*}},
// JSON-NEXT:      "col": 1,
// JSON-NEXT:      "tokLen": 9
// JSON-NEXT:     },
// JSON-NEXT:     "end": {
// JSON-NEXT:      "offset": {{.*}},
// JSON-NEXT:      "line": {{.*}},
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
// JSON-NEXT:       "offset": {{.*}},
// JSON-NEXT:       "line": {{.*}},
// JSON-NEXT:       "col": 41,
// JSON-NEXT:       "tokLen": 9
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "col": 3,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "col": 19,
// JSON-NEXT:         "tokLen": 3
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 13,
// JSON-NEXT:          "tokLen": 5
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "col": 29,
// JSON-NEXT:         "tokLen": 3
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 24,
// JSON-NEXT:          "tokLen": 4
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "col": 41,
// JSON-NEXT:         "tokLen": 9
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 34,
// JSON-NEXT:          "tokLen": 6
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:           "offset": {{.*}},
// JSON-NEXT:           "col": 41,
// JSON-NEXT:           "tokLen": 9
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 34,
// JSON-NEXT:            "tokLen": 6
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
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
// JSON-NEXT:       "offset": {{.*}},
// JSON-NEXT:       "line": {{.*}},
// JSON-NEXT:       "col": 41,
// JSON-NEXT:       "tokLen": 9
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "col": 3,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "col": 19,
// JSON-NEXT:         "tokLen": 3
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 13,
// JSON-NEXT:          "tokLen": 5
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "col": 29,
// JSON-NEXT:         "tokLen": 3
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 24,
// JSON-NEXT:          "tokLen": 4
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "col": 41,
// JSON-NEXT:         "tokLen": 9
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 34,
// JSON-NEXT:          "tokLen": 6
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:           "offset": {{.*}},
// JSON-NEXT:           "col": 41,
// JSON-NEXT:           "tokLen": 9
// JSON-NEXT:          },
// JSON-NEXT:          "range": {
// JSON-NEXT:           "begin": {
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 34,
// JSON-NEXT:            "tokLen": 6
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
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
// JSON-NEXT:       "offset": {{.*}},
// JSON-NEXT:       "line": {{.*}},
// JSON-NEXT:       "col": 10,
// JSON-NEXT:       "tokLen": 9
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "line": {{.*}},
// JSON-NEXT:        "col": 3,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "line": {{.*}},
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
// JSON-NEXT:         "qualType": "Template1<type-parameter-0-0, value-parameter-0-1>"
// JSON-NEXT:        },
// JSON-NEXT:        "typeDetails": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "TemplateSpecializationType",
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "Template1<type-parameter-0-0, value-parameter-0-1>"
// JSON-NEXT:          },
// JSON-NEXT:          "isDependent": true,
// JSON-NEXT:          "isInstantiationDependent": true,
// JSON-NEXT:          "templateName": "TestPartialSpecNTTP::Template1",
// JSON-NEXT:          "qualDetails": [],
// JSON-NEXT:          "inner": [
// JSON-NEXT:           {
// JSON-NEXT:            "kind": "TemplateArgument",
// JSON-NEXT:            "type": {
// JSON-NEXT:             "qualType": "type-parameter-0-0"
// JSON-NEXT:            },
// JSON-NEXT:            "typeDetails": [
// JSON-NEXT:             {
// JSON-NEXT:              "depth": 0,
// JSON-NEXT:              "index": 0,
// JSON-NEXT:              "decl": {
// JSON-NEXT:               "id": "{{.*}}"
// JSON-NEXT:              },
// JSON-NEXT:              "qualDetails": [],
// JSON-NEXT:              "typeDetails": [
// JSON-NEXT:               {
// JSON-NEXT:                "refId": "0x{{.*}}"
// JSON-NEXT:               }
// JSON-NEXT:              ]
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
// JSON-NEXT:                "offset": {{.*}},
// JSON-NEXT:                "line": {{.*}},
// JSON-NEXT:                "col": 29,
// JSON-NEXT:                "tokLen": 3
// JSON-NEXT:               },
// JSON-NEXT:               "end": {
// JSON-NEXT:                "offset": {{.*}},
// JSON-NEXT:                "col": 29,
// JSON-NEXT:                "tokLen": 3
// JSON-NEXT:               }
// JSON-NEXT:              },
// JSON-NEXT:              "type": {
// JSON-NEXT:               "qualType": "bool"
// JSON-NEXT:              },
// JSON-NEXT:              "valueCategory": "prvalue",
// JSON-NEXT:              "referencedDecl": {
// JSON-NEXT:               "id": "0x{{.*}}",
// JSON-NEXT:               "kind": "NonTypeTemplateParmDecl",
// JSON-NEXT:               "name": "TA2",
// JSON-NEXT:               "type": {
// JSON-NEXT:                "qualType": "bool"
// JSON-NEXT:               }
// JSON-NEXT:              },
// JSON-NEXT:              "qualType": {
// JSON-NEXT:               "refId": "0x{{.*}}",
// JSON-NEXT:               "qualDetails": [
// JSON-NEXT:                "unsigned",
// JSON-NEXT:                "integer"
// JSON-NEXT:               ]
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
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "line": {{.*}},
// JSON-NEXT:            "col": 39,
// JSON-NEXT:            "tokLen": 2
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
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
// JSON-NEXT:          },
// JSON-NEXT:          "qualType": {
// JSON-NEXT:           "refId": "0x{{.*}}",
// JSON-NEXT:           "qualDetails": [
// JSON-NEXT:            "unsigned",
// JSON-NEXT:            "integer"
// JSON-NEXT:           ]
// JSON-NEXT:          }
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "TemplateTypeParmDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "line": {{.*}},
// JSON-NEXT:         "col": 19,
// JSON-NEXT:         "tokLen": 2
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 13,
// JSON-NEXT:          "tokLen": 5
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "col": 28,
// JSON-NEXT:         "tokLen": 2
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 23,
// JSON-NEXT:          "tokLen": 4
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "col": 37,
// JSON-NEXT:         "tokLen": 2
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 32,
// JSON-NEXT:          "tokLen": 4
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "line": {{.*}},
// JSON-NEXT:         "col": 10,
// JSON-NEXT:         "tokLen": 9
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 3,
// JSON-NEXT:          "tokLen": 6
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:       "offset": {{.*}},
// JSON-NEXT:       "line": {{.*}},
// JSON-NEXT:       "col": 10,
// JSON-NEXT:       "tokLen": 9
// JSON-NEXT:      },
// JSON-NEXT:      "range": {
// JSON-NEXT:       "begin": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "line": {{.*}},
// JSON-NEXT:        "col": 3,
// JSON-NEXT:        "tokLen": 8
// JSON-NEXT:       },
// JSON-NEXT:       "end": {
// JSON-NEXT:        "offset": {{.*}},
// JSON-NEXT:        "line": {{.*}},
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
// JSON-NEXT:         "qualType": "Template1<type-parameter-0-0, value-parameter-0-2>"
// JSON-NEXT:        },
// JSON-NEXT:        "typeDetails": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "TemplateSpecializationType",
// JSON-NEXT:          "type": {
// JSON-NEXT:           "qualType": "Template1<type-parameter-0-0, value-parameter-0-2>"
// JSON-NEXT:          },
// JSON-NEXT:          "isDependent": true,
// JSON-NEXT:          "isInstantiationDependent": true,
// JSON-NEXT:          "templateName": "TestPartialSpecNTTP::Template1",
// JSON-NEXT:          "qualDetails": [],
// JSON-NEXT:          "inner": [
// JSON-NEXT:           {
// JSON-NEXT:            "kind": "TemplateArgument",
// JSON-NEXT:            "type": {
// JSON-NEXT:             "qualType": "type-parameter-0-0"
// JSON-NEXT:            },
// JSON-NEXT:            "typeDetails": [
// JSON-NEXT:             {
// JSON-NEXT:              "depth": 0,
// JSON-NEXT:              "index": 0,
// JSON-NEXT:              "decl": {
// JSON-NEXT:               "id": "{{.*}}"
// JSON-NEXT:              },
// JSON-NEXT:              "qualDetails": [],
// JSON-NEXT:              "typeDetails": [
// JSON-NEXT:               {
// JSON-NEXT:                "refId": "0x{{.*}}"
// JSON-NEXT:               }
// JSON-NEXT:              ]
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
// JSON-NEXT:                "offset": {{.*}},
// JSON-NEXT:                "col": 34,
// JSON-NEXT:                "tokLen": 2
// JSON-NEXT:               },
// JSON-NEXT:               "end": {
// JSON-NEXT:                "offset": {{.*}},
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
// JSON-NEXT:              },
// JSON-NEXT:              "qualType": {
// JSON-NEXT:               "refId": "0x{{.*}}",
// JSON-NEXT:               "qualDetails": [
// JSON-NEXT:                "unsigned",
// JSON-NEXT:                "integer"
// JSON-NEXT:               ]
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
// JSON-NEXT:            "offset": {{.*}},
// JSON-NEXT:            "col": 39,
// JSON-NEXT:            "tokLen": 2
// JSON-NEXT:           },
// JSON-NEXT:           "end": {
// JSON-NEXT:            "offset": {{.*}},
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
// JSON-NEXT:          },
// JSON-NEXT:          "qualType": {
// JSON-NEXT:           "refId": "0x{{.*}}",
// JSON-NEXT:           "qualDetails": [
// JSON-NEXT:            "unsigned",
// JSON-NEXT:            "integer"
// JSON-NEXT:           ]
// JSON-NEXT:          }
// JSON-NEXT:         }
// JSON-NEXT:        ]
// JSON-NEXT:       },
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "TemplateTypeParmDecl",
// JSON-NEXT:        "loc": {
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "line": {{.*}},
// JSON-NEXT:         "col": 22,
// JSON-NEXT:         "tokLen": 2
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 13,
// JSON-NEXT:          "tokLen": 8
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "col": 31,
// JSON-NEXT:         "tokLen": 2
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 26,
// JSON-NEXT:          "tokLen": 4
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "col": 40,
// JSON-NEXT:         "tokLen": 2
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 35,
// JSON-NEXT:          "tokLen": 4
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:         "offset": {{.*}},
// JSON-NEXT:         "line": {{.*}},
// JSON-NEXT:         "col": 10,
// JSON-NEXT:         "tokLen": 9
// JSON-NEXT:        },
// JSON-NEXT:        "range": {
// JSON-NEXT:         "begin": {
// JSON-NEXT:          "offset": {{.*}},
// JSON-NEXT:          "col": 3,
// JSON-NEXT:          "tokLen": 6
// JSON-NEXT:         },
// JSON-NEXT:         "end": {
// JSON-NEXT:          "offset": {{.*}},
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
// JSON-NEXT:   }
// JSON-NEXT:  ]
// JSON-NEXT: }
