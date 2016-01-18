// RUN: %clang_cc1 -std=c++11 -ast-dump -ast-dump-filter Test %s | FileCheck -check-prefix DECLS %s
// RUN: %clang_cc1 -std=c++11 -ast-dump-lookups -ast-dump-filter Test %s | FileCheck -check-prefix LOOKUPS %s
// RUN: %clang_cc1 -std=c++11 -ast-dump -ast-dump-lookups -ast-dump-filter Test %s | FileCheck -check-prefix DECLS-LOOKUPS %s

namespace Test {
  extern int a;
  int a = 0;
}

namespace Test { }

// DECLS: Dumping Test:
// DECLS-NEXT: NamespaceDecl {{.*}} Test
// DECLS-NEXT: |-VarDecl [[EXTERN_A:0x[^ ]*]] {{.*}} a 'int' extern
// DECLS-NEXT: `-VarDecl {{.*}} prev [[EXTERN_A]] {{.*}} a 'int' cinit
// DECLS-NEXT:   `-IntegerLiteral {{.*}} 'int' 0
//
// DECLS: Dumping Test:
// DECLS-NEXT: NamespaceDecl {{.*}} Test

// LOOKUPS: Dumping Test:
// LOOKUPS-NEXT: StoredDeclsMap Namespace {{.*}} 'Test'
// LOOKUPS-NEXT: `-DeclarationName 'a'
// LOOKUPS-NEXT:   `-Var {{.*}} 'a' 'int'
//
// LOOKUPS: Dumping Test:
// LOOKUPS-NEXT: Lookup map is in primary DeclContext

// DECLS-LOOKUPS: Dumping Test:
// DECLS-LOOKUPS-NEXT: StoredDeclsMap Namespace {{.*}} 'Test'
// DECLS-LOOKUPS-NEXT: `-DeclarationName 'a'
// DECLS-LOOKUPS-NEXT:   `-Var [[A:[^ ]*]] 'a' 'int'
// DECLS-LOOKUPS-NEXT:     |-VarDecl [[EXTERN_A:0x[^ ]*]] {{.*}} a 'int' extern
// DECLS-LOOKUPS-NEXT:     `-VarDecl [[A]] prev [[EXTERN_A]] {{.*}} a 'int' cinit
// DECLS-LOOKUPS-NEXT:       `-IntegerLiteral {{.*}} 'int' 0
//
// DECLS-LOOKUPS: Dumping Test:
// DECLS-LOOKUPS-NEXT: Lookup map is in primary DeclContext
