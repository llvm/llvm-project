// NOLINTNEXTLINE
class A { A(int i); };

class B { B(int i); };
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: single-argument constructors must be marked explicit

// NOLINTNEXTLINE
class C { C(int i); };

// NOLINTNEXTLINE(for-some-other-check)
class D { D(int i); };
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: single-argument constructors must be marked explicit

// NOLINTNEXTLINE()
class D1 { D1(int i); };
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: single-argument constructors must be marked explicit

// NOLINTNEXTLINE(*)
class D2 { D2(int i); };

// NOLINTNEXTLINE(not-closed-bracket-is-treated-as-skip-all
class D3 { D3(int i); };

// NOLINTNEXTLINE(google-explicit-constructor)
class D4 { D4(int i); };

// NOLINTNEXTLINE(some-check, google-explicit-constructor)
class D5 { D5(int i); };

// NOLINTNEXTLINE without-brackets-skip-all, another-check
class D6 { D6(int i); };

// NOLINTNEXTLINE

class E { E(int i); };
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: single-argument constructors must be marked explicit

// NOLINTNEXTLINE
//
class F { F(int i); };
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: single-argument constructors must be marked explicit

#define MACRO(X) class X { X(int i); };
MACRO(G)
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: single-argument constructors must be marked explicit
// NOLINTNEXTLINE
MACRO(H)

#define MACRO_NOARG class I { I(int i); };
// NOLINTNEXTLINE
MACRO_NOARG

// NOLINTNEXTLINE(google*)
class I1 { I1(int i); };

// NOLINTNEXTLINE(*explicit-constructor)
class I2 { I2(int i); };

// NOLINTNEXTLINE(*explicit*)
class I3 { I3(int i); };

// NOLINTNEXTLINE(-explicit-constructor)
class I4 { I4(int x); };
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: single-argument constructors must be marked explicit

// NOLINTNEXTLINE(google*,-google*)
class I5 { I5(int x); };

// NOLINTNEXTLINE(*,-google*)
class I6 { I6(int x); };

int array1[10];
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: do not declare C-style arrays, use 'std::array' instead [cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays]

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays)
int array2[10];
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: do not declare C-style arrays, use 'std::array' instead [modernize-avoid-c-arrays]

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
int array3[10];

// NOLINTNEXTLINE(*-avoid-c-arrays)
int array4[10];

// CHECK-MESSAGES: Suppressed 19 warnings (19 NOLINT)

// RUN: %check_clang_tidy %s google-explicit-constructor,clang-diagnostic-unused-variable,cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays %t -- -extra-arg=-Wunused-variable
