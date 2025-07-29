// RUN: not %clang -fdiagnostics-format=fancy -fsyntax-only -ferror-limit=0 %s 2>&1 | FileCheck %s --strict-whitespace

#define FOO BAR
#define BAR BAZ
#define BAZ undefined

struct X { int x; };
struct Y { Y(const Y&); };

void g() {
  FOO
}

static_assert(__is_assignable(int&, void));
static_assert(__is_empty(X&));
static_assert(__is_trivially_copyable(Y));

#include "Inputs/a.h"

// CHECK: error: use of undeclared identifier 'undefined'
// CHECK-NEXT: |
// CHECK-NEXT: |     - at {{.*}}fancy-diags-format.cpp:11:3:
// CHECK-NEXT: |
// CHECK-NEXT: |  11 |   FOO
// CHECK-NEXT: |     |   ^~~
// CHECK-NEXT: |
// CHECK-NEXT: |-- note: expanded from macro 'FOO'
// CHECK-NEXT: |
// CHECK-NEXT: |         - at {{.*}}fancy-diags-format.cpp:3:13:
// CHECK-NEXT: |
// CHECK-NEXT: |       3 | #define FOO BAR
// CHECK-NEXT: |         |             ^~~
// CHECK-NEXT: |
// CHECK-NEXT: |-- note: expanded from macro 'BAR'
// CHECK-NEXT: |
// CHECK-NEXT: |         - at {{.*}}fancy-diags-format.cpp:4:13:
// CHECK-NEXT: |
// CHECK-NEXT: |       4 | #define BAR BAZ
// CHECK-NEXT: |         |             ^~~
// CHECK-NEXT: |
// CHECK-NEXT: |-- note: expanded from macro 'BAZ'
// CHECK-NEXT: |
// CHECK-NEXT: |         - at {{.*}}fancy-diags-format.cpp:5:13:
// CHECK-NEXT: |
// CHECK-NEXT: |       5 | #define BAZ undefined
// CHECK-NEXT: |         |             ^~~~~~~~~
// CHECK-EMPTY:

// CHECK: error: static assertion failed due to requirement '__is_assignable(int &, void)'
// CHECK-NEXT: |
// CHECK-NEXT: |     - at {{.*}}fancy-diags-format.cpp:14:15:
// CHECK-NEXT: |
// CHECK-NEXT: |  14 | static_assert(__is_assignable(int&, void));
// CHECK-NEXT: |     |               ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK-NEXT: |
// CHECK-NEXT: |-- error: assigning to 'int' from incompatible type 'void'
// CHECK-NEXT: |
// CHECK-NEXT: |         - at {{.*}}fancy-diags-format.cpp:14:15:
// CHECK-NEXT: |
// CHECK-NEXT: |      14 | static_assert(__is_assignable(int&, void));
// CHECK-NEXT: |         |               ^~~~~~~~~~~~~~~
// CHECK-EMPTY:

// CHECK: error: static assertion failed due to requirement '__is_empty(X &)'
// CHECK-NEXT: |
// CHECK-NEXT: |     - at {{.*}}fancy-diags-format.cpp:15:15:
// CHECK-NEXT: |
// CHECK-NEXT: |  15 | static_assert(__is_empty(X&));
// CHECK-NEXT: |     |               ^~~~~~~~~~~~~~
// CHECK-NEXT: |
// CHECK-NEXT: |-- note: 'X &' is not empty
// CHECK-NEXT: |
// CHECK-NEXT: |         - at {{.*}}fancy-diags-format.cpp:15:15:
// CHECK-NEXT: |
// CHECK-NEXT: |
// CHECK-NEXT: |------ note: because it is a reference type
// CHECK-NEXT: |
// CHECK-NEXT: |             - at {{.*}}fancy-diags-format.cpp:15:15:
// CHECK-NEXT: |
// CHECK-NEXT: |
// CHECK-NEXT: |------ note: because it has a non-static data member 'x' of type 'int'
// CHECK-NEXT: |
// CHECK-NEXT: |             - at {{.*}}fancy-diags-format.cpp:15:15:
// CHECK-NEXT: |
// CHECK-NEXT: |           7 | struct X { int x; };
// CHECK-NEXT: |             |            ~~~~~
// CHECK-NEXT: |           8 | struct Y { Y(const Y&); };
// CHECK-NEXT: |           9 |
// CHECK-NEXT: |          10 | void g() {
// CHECK-NEXT: |          11 |   FOO
// CHECK-NEXT: |          12 | }
// CHECK-NEXT: |          13 |
// CHECK-NEXT: |          14 | static_assert(__is_assignable(int&, void));
// CHECK-NEXT: |          15 | static_assert(__is_empty(X&));
// CHECK-NEXT: |             |               ^
// CHECK-NEXT: |
// CHECK-NEXT: |---------- note: 'X' defined here
// CHECK-NEXT: |
// CHECK-NEXT: |                 - at {{.*}}fancy-diags-format.cpp:7:8:
// CHECK-NEXT: |
// CHECK-NEXT: |               7 | struct X { int x; };
// CHECK-NEXT: |                 |        ^
// CHECK-EMPTY:

// CHECK: error: static assertion failed due to requirement '__is_trivially_copyable(Y)'
// CHECK-NEXT: |
// CHECK-NEXT: |     - at {{.*}}fancy-diags-format.cpp:16:15:
// CHECK-NEXT: |
// CHECK-NEXT: |  16 | static_assert(__is_trivially_copyable(Y));
// CHECK-NEXT: |     |               ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK-NEXT: |
// CHECK-NEXT: |-- note: 'Y' is not trivially copyable
// CHECK-NEXT: |
// CHECK-NEXT: |         - at {{.*}}fancy-diags-format.cpp:16:15:
// CHECK-NEXT: |
// CHECK-NEXT: |
// CHECK-NEXT: |------ note: because it has a user provided copy constructor
// CHECK-NEXT: |
// CHECK-NEXT: |             - at {{.*}}fancy-diags-format.cpp:16:15:
// CHECK-NEXT: |
// CHECK-NEXT: |           8 | struct Y { Y(const Y&); };
// CHECK-NEXT: |             |            ~~~~~~~~~~~
// CHECK-NEXT: |           9 |
// CHECK-NEXT: |          10 | void g() {
// CHECK-NEXT: |          11 |   FOO
// CHECK-NEXT: |          12 | }
// CHECK-NEXT: |          13 |
// CHECK-NEXT: |          14 | static_assert(__is_assignable(int&, void));
// CHECK-NEXT: |          15 | static_assert(__is_empty(X&));
// CHECK-NEXT: |          16 | static_assert(__is_trivially_copyable(Y));
// CHECK-NEXT: |             |               ^
// CHECK-NEXT: |
// CHECK-NEXT: |---------- note: 'Y' defined here
// CHECK-NEXT: |
// CHECK-NEXT: |                 - at {{.*}}fancy-diags-format.cpp:8:8:
// CHECK-NEXT: |
// CHECK-NEXT: |               8 | struct Y { Y(const Y&); };
// CHECK-NEXT: |                 |        ^
// CHECK-EMPTY:

// CHECK: error: a type specifier is required for all declarations
// CHECK-NEXT: |
// CHECK-NEXT: |     - included from {{.*}}fancy-diags-format.cpp:18:10:
// CHECK-NEXT: |     - included from {{.*}}a.h:1:10:
// CHECK-NEXT: |     - included from {{.*}}b.h:1:10:
// CHECK-NEXT: |     - at {{.*}}c.h:1:1:
// CHECK-NEXT: |
// CHECK-NEXT: |   1 | undefined_c;
// CHECK-NEXT: |     | ^
// CHECK-EMPTY:

// CHECK: error: a type specifier is required for all declarations
// CHECK-NEXT: |
// CHECK-NEXT: |     - included from {{.*}}fancy-diags-format.cpp:18:10:
// CHECK-NEXT: |     - included from {{.*}}a.h:1:10:
// CHECK-NEXT: |     - at {{.*}}b.h:3:1:
// CHECK-NEXT: |
// CHECK-NEXT: |   3 | undefined_b;
// CHECK-NEXT: |     | ^
// CHECK-EMPTY:

// CHECK: error: a type specifier is required for all declarations
// CHECK-NEXT: |
// CHECK-NEXT: |     - included from {{.*}}fancy-diags-format.cpp:18:10:
// CHECK-NEXT: |     - at {{.*}}a.h:3:1:
// CHECK-NEXT: |
// CHECK-NEXT: |   3 | undefined_a;
// CHECK-NEXT: |     | ^

// CHECK-NEXT: {{.*}} errors generated.
