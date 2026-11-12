// RUN: %clang_cc1 -w -triple=x86_64-pc-win32 -fms-compatibility -fdump-record-layouts -foverride-record-layout=%S/Inputs/override-layout-ms.layout %s | FileCheck  %s
// RUN: %clang_cc1 -w -triple=x86_64-pc-win32 -fms-compatibility -fdump-record-layouts %s | FileCheck  %s

// CHECK: *** Dumping AST Record Layout
// CHECK:          0 | struct E1 (empty)
// CHECK:            | [sizeof=1, align=1,
// CHECK:            |  nvsize=0, nvalign=1]
// CHECK: *** Dumping AST Record Layout
// CHECK:          0 | struct Mid
// CHECK:          0 |   void * p
// CHECK:            | [sizeof=8, align=8,
// CHECK:            |  nvsize=8, nvalign=8]
// CHECK: *** Dumping AST Record Layout
// CHECK:          0 | struct E2 (empty)
// CHECK:            | [sizeof=1, align=1,
// CHECK:            |  nvsize=0, nvalign=1]
// CHECK: *** Dumping AST Record Layout
// CHECK:          0 | struct Combine
// CHECK:          0 |   struct E1 (base) (empty)
// CHECK:          0 |   struct Mid (base)
// CHECK:          0 |     void * p
// CHECK:          0 |   struct E2 (base) (empty)
// CHECK:            | [sizeof=8, align=8,
// CHECK:            |  nvsize=8, nvalign=8]
// CHECK: *** Dumping AST Record Layout
// CHECK:          0 | struct Combine2
// CHECK:          0 |   struct VB1 (primary base)
// CHECK:          0 |     (VB1 vftable pointer)
// CHECK:          8 |   struct VB2 (base)
// CHECK:          8 |     (VB2 vftable pointer)
// CHECK:            | [sizeof=16, align=8,
// CHECK:            |  nvsize=16, nvalign=8]


struct E1 {};
struct E2 {};
struct Mid {void *p; };
struct __declspec(empty_bases) Combine : E1, Mid, E2 {};
struct VB1 { virtual void foo() {}};
struct VB2 { virtual void bar() {}};
struct Combine2: VB1, VB2 {};
Combine g;
Combine2 f;