// RUN: %clang_cc1 -fno-rtti -triple i686-pc-win32 -fdump-record-layouts -fms-extensions -fsyntax-only %s 2>&1 | FileCheck %s

struct __single_inheritance S;
struct __multiple_inheritance M;
struct __virtual_inheritance V;
struct U;

struct SD { char a; int S::*mp; };
struct MD { char a; int M::*mp; };
struct VD { char a; int V::*mp; };
struct UD { char a; int U::*mp; };
struct SF { char a; int (S::*mp)(); };
struct MF { char a; int (M::*mp)(); };
struct VF { char a; int (V::*mp)(); };
struct UF { char a; int (U::*mp)(); };

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:    0 | struct SD
// CHECK-NEXT:    0 |   char a
// CHECK-NEXT:    4 |   int S::* mp
// CHECK-NEXT:      | [sizeof=8, align=4
// CHECK-NEXT:      |  nvsize=8, nvalign=4]

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:    0 | struct MD
// CHECK-NEXT:    0 |   char a
// CHECK-NEXT:    4 |   int M::* mp
// CHECK-NEXT:      | [sizeof=8, align=4
// CHECK-NEXT:      |  nvsize=8, nvalign=4]

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:    0 | struct VD
// CHECK-NEXT:    0 |   char a
// CHECK-NEXT:    8 |   int V::* mp
// CHECK-NEXT:      | [sizeof=16, align=8
// CHECK-NEXT:      |  nvsize=16, nvalign=8]

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:    0 | struct UD
// CHECK-NEXT:    0 |   char a
// CHECK-NEXT:    8 |   int U::* mp
// CHECK-NEXT:      | [sizeof=24, align=8
// CHECK-NEXT:      |  nvsize=24, nvalign=8]

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:    0 | struct SF
// CHECK-NEXT:    0 |   char a
// CHECK-NEXT:    4 |   int (S::*)(void) __attribute__((thiscall)) mp
// CHECK-NEXT:      | [sizeof=8, align=4
// CHECK-NEXT:      |  nvsize=8, nvalign=4]

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:    0 | struct MF
// CHECK-NEXT:    0 |   char a
// CHECK-NEXT:    8 |   int (M::*)(void) __attribute__((thiscall)) mp
// CHECK-NEXT:      | [sizeof=16, align=8
// CHECK-NEXT:      |  nvsize=16, nvalign=8]

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:    0 | struct VF
// CHECK-NEXT:    0 |   char a
// CHECK-NEXT:    8 |   int (V::*)(void) __attribute__((thiscall)) mp
// CHECK-NEXT:      | [sizeof=24, align=8
// CHECK-NEXT:      |  nvsize=24, nvalign=8]

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:    0 | struct UF
// CHECK-NEXT:    0 |   char a
// CHECK-NEXT:    8 |   int (U::*)(void) __attribute__((thiscall)) mp
// CHECK-NEXT:      | [sizeof=24, align=8
// CHECK-NEXT:      |  nvsize=24, nvalign=8]

char a[sizeof(SD) +
       sizeof(MD) +
       sizeof(VD) +
       sizeof(UD) +
       sizeof(SF) +
       sizeof(MF) +
       sizeof(VF) +
       sizeof(UF)];
