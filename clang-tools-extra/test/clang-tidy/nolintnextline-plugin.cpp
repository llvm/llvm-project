// RUN: c-index-test -test-load-source-reparse 2 all %s -Xclang -add-plugin -Xclang clang-tidy -Xclang -plugin-arg-clang-tidy -Xclang -checks='-*,google-explicit-constructor' 2>&1 | FileCheck %s

class A { A(int i); };
// CHECK: :[[@LINE-1]]:11: warning: single-argument constructors must be marked explicit

// NOLINTNEXTLINE
class B { B(int i); };

// NOLINTNEXTLINE(for-some-other-check)
class C { C(int i); };
// CHECK: :[[@LINE-1]]:11: warning: single-argument constructors must be marked explicit

// NOLINTNEXTLINE(*)
class C1 { C1(int i); };

// NOLINTNEXTLINE(not-closed-bracket-is-treated-as-skip-all
class C2 { C2(int i); };

// NOLINTNEXTLINE(google-explicit-constructor)
class C3 { C3(int i); };

// NOLINTNEXTLINE(some-check, google-explicit-constructor)
class C4 { C4(int i); };

// NOLINTNEXTLINE without-brackets-skip-all, another-check
class C5 { C5(int i); };


// NOLINTNEXTLINE

class D { D(int i); };
// CHECK: :[[@LINE-1]]:11: warning: single-argument constructors must be marked explicit

// NOLINTNEXTLINE
//
class E { E(int i); };
// CHECK: :[[@LINE-1]]:11: warning: single-argument constructors must be marked explicit

#define MACRO(X) class X { X(int i); };
MACRO(F)
// CHECK: :[[@LINE-1]]:7: warning: single-argument constructors must be marked explicit
// NOLINTNEXTLINE
MACRO(G)

#define MACRO_NOARG class H { H(int i); };
// NOLINTNEXTLINE
MACRO_NOARG

