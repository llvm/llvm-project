// RUN: %clang_cc1 -fsyntax-only -ast-dump -std=c++2c %s | FileCheck %s
// RUN: %clang_cc1 -ast-print -std=c++2c %s | FileCheck %s --check-prefix=PRINT
// RUN: %clang_cc1 -emit-pch -std=c++2c -o %t %s
// RUN: %clang_cc1 -x c++ -std=c++2c -include-pch %t -ast-dump-all /dev/null

struct S;
template <typename> struct TS; // #template

// CHECK-LABEL: CXXRecordDecl {{.*}} struct Friends
// PRINT-LABEL: struct Friends {
struct Friends {
  // CHECK: FriendDecl {{.*}} 'int'
  // CHECK-NEXT: FriendDecl {{.*}} 'long'
  // PRINT-NEXT: friend int;
  // PRINT-NEXT: friend long;
  friend int, long;

  // CHECK-NEXT: FriendDecl {{.*}} 'int'
  // CHECK-NEXT: FriendDecl {{.*}} 'long'
  // CHECK-NEXT: FriendDecl {{.*}} 'char'
  // PRINT-NEXT: friend int;
  // PRINT-NEXT: friend long;
  // PRINT-NEXT: friend char;
  friend int, long, char;

  // CHECK-NEXT: FriendDecl {{.*}} 'S'
  // PRINT-NEXT: friend S;
  friend S;

  // CHECK-NEXT: FriendDecl {{.*}} 'S'
  // CHECK-NEXT: FriendDecl {{.*}} 'S'
  // CHECK-NEXT: FriendDecl {{.*}} 'S'
  // PRINT-NEXT: friend S;
  // PRINT-NEXT: friend S;
  // PRINT-NEXT: friend S;
  friend S, S, S;

  // CHECK-NEXT: FriendDecl
  // CHECK-NEXT: ClassTemplateDecl {{.*}} friend TS
  // PRINT-NEXT: friend template <typename> struct TS;
  template <typename> friend struct TS;
};

namespace specialisations {
template<class T>
struct C {
  template<class U> struct Nested;
};

struct N {
  template<class U> class C;
};

// CHECK-LABEL: ClassTemplateDecl {{.*}} Variadic
// PRINT-LABEL: template <typename ...Pack> struct Variadic {
template <typename ...Pack> struct Variadic {
  // CHECK: FriendDecl {{.*}} 'Pack'...
  // CHECK-NEXT: FriendDecl {{.*}} 'long'
  // CHECK-NEXT: FriendDecl {{.*}} 'Pack'...
  // PRINT-NEXT: friend Pack...;
  // PRINT-NEXT: friend long;
  // PRINT-NEXT: friend Pack...;
  friend Pack..., long, Pack...;

  // CHECK-NEXT: FriendDecl {{.*}} 'TS<Pack>'...
  // PRINT-NEXT: friend TS<Pack>...;
  friend TS<Pack>...;
};

// CHECK-LABEL: ClassTemplateDecl {{.*}} S2
// PRINT-LABEL: template <class ...Ts> struct S2 {
template<class ...Ts> struct S2 {
  // CHECK: FriendDecl {{.*}} 'class C<Ts>':'C<Ts>'...
  // PRINT-NEXT: friend class C<Ts>...;
  friend class C<Ts>...;

  // CHECK-NEXT: FriendDecl {{.*}} 'class N::C<Ts>':'C<Ts>'...
  // PRINT-NEXT: friend class N::C<Ts>...
  friend class N::C<Ts>...;
};
}
