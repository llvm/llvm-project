// RUN: %clang_cc1 -std=c++11 -ast-dump -ast-dump-filter Test %s | FileCheck -check-prefix DECLS %s
// RUN: %clang_cc1 -std=c++11 -ast-dump-lookups -ast-dump-filter Test %s | FileCheck -check-prefix LOOKUPS %s
// RUN: %clang_cc1 -std=c++11 -ast-dump -ast-dump-lookups -ast-dump-filter Test %s | FileCheck -check-prefix DECLS-LOOKUPS %s
// RUN: %clang_cc1 -std=c++11 -DPRAGMA -fsyntax-only -verify %s 2>&1 | FileCheck -check-prefix PRAGMA %s

namespace Test {
  typedef int T;
  extern int a;
  int a = 0;
}

#ifdef PRAGMA
#pragma clang __debug dump Test
// PRAGMA: lookup results for Test:
// PRAGMA-NEXT: NamespaceDecl {{.*}} Test
// PRAGMA-NEXT: |-TypedefDecl {{.*}} T 'int'
// PRAGMA-NEXT: | `-BuiltinType {{.*}} 'int'
// PRAGMA-NEXT: |-VarDecl [[EXTERN_A:0x[^ ]*]] {{.*}} a 'int' extern
// PRAGMA-NEXT: `-VarDecl {{.*}} prev [[EXTERN_A]] {{.*}} a 'int' cinit
// PRAGMA-NEXT:   `-IntegerLiteral {{.*}} 'int' 0
#endif

namespace Test { }

// DECLS: Dumping Test:
// DECLS-NEXT: NamespaceDecl {{.*}} Test
// DECLS-NEXT: |-TypedefDecl {{.*}} T 'int'
// DECLS-NEXT: | `-BuiltinType {{.*}} 'int'
// DECLS-NEXT: |-VarDecl [[EXTERN_A:0x[^ ]*]] {{.*}} a 'int' extern
// DECLS-NEXT: `-VarDecl {{.*}} prev [[EXTERN_A]] {{.*}} a 'int' cinit
// DECLS-NEXT:   `-IntegerLiteral {{.*}} 'int' 0
//
// DECLS: Dumping Test:
// DECLS-NEXT: NamespaceDecl {{.*}} Test

// LOOKUPS: Dumping Test:
// LOOKUPS-NEXT: StoredDeclsMap Namespace {{.*}} 'Test'
// LOOKUPS:      DeclarationName 'a'
// LOOKUPS-NEXT: `-Var {{.*}} 'a' 'int'
//
// LOOKUPS: Dumping Test:
// LOOKUPS-NEXT: Lookup map is in primary DeclContext

// DECLS-LOOKUPS: Dumping Test:
// DECLS-LOOKUPS-NEXT: StoredDeclsMap Namespace {{.*}} 'Test'
// DECLS-LOOKUPS:       -DeclarationName 'a'
// DECLS-LOOKUPS-NEXT:   `-Var [[A:[^ ]*]] 'a' 'int'
// DECLS-LOOKUPS-NEXT:     |-VarDecl [[EXTERN_A:0x[^ ]*]] {{.*}} a 'int' extern
// DECLS-LOOKUPS-NEXT:     `-VarDecl [[A]] prev [[EXTERN_A]] {{.*}} a 'int' cinit
// DECLS-LOOKUPS-NEXT:       `-IntegerLiteral {{.*}} 'int' 0
//
// DECLS-LOOKUPS: Dumping Test:
// DECLS-LOOKUPS-NEXT: Lookup map is in primary DeclContext

#ifdef PRAGMA
namespace Test {
  struct S {
    const S& operator+(const S&) { return *this; }
  };
  void foo(S) {}
}

#pragma clang __debug dump foo(Test::S{})
// PRAGMA: CallExpr {{.*}} adl
// PRAGMA-NEXT: |-ImplicitCastExpr {{.*}}
// PRAGMA-NEXT: | `-DeclRefExpr {{.*}} 'void (S)' lvalue Function {{.*}} 'foo' 'void (S)'

#pragma clang __debug dump Test::S{} + Test::S{}
// PRAGMA: CXXOperatorCallExpr {{.*}}
// PRAGMA-NEXT: |-ImplicitCastExpr {{.*}}
// PRAGMA-NEXT: | `-DeclRefExpr {{.*}} 'const S &(const S &)' lvalue CXXMethod {{.*}} 'operator+' 'const S &(const S &)'

#pragma clang __debug dump &Test::S::operator+
// PRAGMA: UnaryOperator {{.*}}
// PRAGMA-NEXT: `-DeclRefExpr {{.*}} 'const S &(const S &)' CXXMethod {{.*}} 'operator+' 'const S &(const S &)'

template<typename T, int I>
void bar() {
#pragma clang __debug dump T{} // expected-warning {{type-dependent expression}}
#pragma clang __debug dump +I  // expected-warning {{value-dependent expression}}
}

template <typename T>
struct S {
  static constexpr const T *str = "string";
};

template <>
struct S<wchar_t> {
  static constexpr const wchar_t *str = L"wide string";
};

void func() {
  #pragma clang __debug dump S<wchar_t>::str;
  // PRAGMA: DeclRefExpr {{.*}} 'const wchar_t *const' lvalue Var {{.*}} 'str' 'const wchar_t *const'
}

#pragma clang __debug dump this is nonsense // expected-error {{invalid use of 'this'}}

#endif
