// RUN: %clang_cc1 -fsyntax-only -triple armv7k-apple-darwin-watchos -fdump-record-layouts %s | FileCheck %s

// WatchOS, 64-bit iOS, and WebAssembly use the C++11 definition of POD to
// determine whether we can reuse the tail padding of a struct (POD is
// "trivially copyable and standard layout"). The definition of standard
// layout changed some time around C++17; check that we still use the old
// ABI rule.

// B is not standard-layout, but it was under C++11's rule, so we pack
// C::d into its tail padding anyway.
struct A { int : 0; };
struct B : A { int n; char c[3]; };
struct C : B { char d; };
int c = sizeof(C);
static_assert(!__is_standard_layout(B));

// CHECK:*** Dumping AST Record Layout
// CHECK:          0 | struct C
// CHECK-NEXT:     0 |   struct B (base)
// CHECK-NEXT:     0 |     struct A (base) (empty)
// CHECK-NEXT:   0:- |       int 
// CHECK-NEXT:     0 |     int n
// CHECK-NEXT:     4 |     char[3] c
// CHECK-NEXT:     8 |   char d
// CHECK-NEXT:       | [sizeof=12, dsize=9, align=4,
// CHECK-NEXT:       |  nvsize=9, nvalign=4]

// F is not standard-layout due to the repeated D base class, but it was under
// C++11's rule, so we pack G::d into its tail padding anyway.
struct D {};
struct E : D {};
struct F : D, E { int n; char c[3]; };
struct G : F { G(const G&); char d; };
int g = sizeof(G);
static_assert(!__is_standard_layout(F));

// CHECK:*** Dumping AST Record Layout
// CHECK:          0 | struct G
// CHECK-NEXT:     0 |   struct F (base)
// CHECK-NEXT:     0 |     struct D (base) (empty)
// CHECK-NEXT:     1 |     struct E (base) (empty)
// CHECK-NEXT:     1 |       struct D (base) (empty)
// CHECK-NEXT:     0 |     int n
// CHECK-NEXT:     4 |     char[3] c
// CHECK-NEXT:     8 |   char d
// CHECK-NEXT:       | [sizeof=12, dsize=9, align=4,
// CHECK-NEXT:       |  nvsize=9, nvalign=4]
