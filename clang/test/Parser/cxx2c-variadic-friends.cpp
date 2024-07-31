// RUN: %clang_cc1 -fsyntax-only -verify -std=c++2c %s
// RUN: %clang_cc1 -fsyntax-only -ast-dump -std=c++2c %s | FileCheck %s
// expected-no-diagnostics

struct S;
template <typename> struct TS; // #template

// CHECK-LABEL: CXXRecordDecl {{.*}} struct Friends
struct Friends {
  // CHECK: FriendDecl {{.*}} 'int'
  // CHECK-NEXT: FriendDecl {{.*}} 'long'
  friend int, long;

  // CHECK-NEXT: FriendDecl {{.*}} 'int'
  // CHECK-NEXT: FriendDecl {{.*}} 'long'
  // CHECK-NEXT: FriendDecl {{.*}} 'char'
  friend int, long, char;

  // CHECK-NEXT: FriendDecl {{.*}} 'S'
  friend S;

  // CHECK-NEXT: FriendDecl {{.*}} 'S'
  // CHECK-NEXT: FriendDecl {{.*}} 'S'
  // CHECK-NEXT: FriendDecl {{.*}} 'S'
  friend S, S, S;

  // CHECK-NEXT: FriendDecl
  // CHECK-NEXT: ClassTemplateDecl {{.*}} friend TS
  template <typename>
  friend struct TS;
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
// CHECK: FriendDecl {{.*}} 'Pack' variadic
// CHECK-NEXT: FriendDecl {{.*}} 'TS<Pack>' variadic
template <typename ...Pack>
struct Variadic {
  friend Pack...;
  friend TS<Pack>...;
};

// CHECK-LABEL: ClassTemplateDecl {{.*}} S2
// CHECK: FriendDecl {{.*}} 'class C<Ts>':'C<Ts>' variadic
// CHECK-NEXT: FriendDecl {{.*}} 'class N::C<Ts>':'C<Ts>' variadic
template<class... Ts>
struct S2 {
  friend class C<Ts>...;
  friend class N::C<Ts>...;
};
}
