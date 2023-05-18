// RUN: %clang_cc1 -E %s | FileCheck --strict-whitespace %s
// RUN: %clang_cc1 -E -P -fminimize-whitespace %s | FileCheck --strict-whitespace %s --check-prefix=MINWS

#define FOO1()
#define FOO2(x)x
#define FOO3(x) x
#define FOO4(x)x x
#define FOO5(x) x x
#define FOO6(x) [x]
#define FOO7(x) [ x]
#define FOO8(x) [x ]

#define TEST(FOO,x) FOO <FOO()> < FOO()> <FOO ()> <FOO( )> <FOO() > <FOO()x> <FOO() x> < FOO()x>

TEST(FOO1,)
// CHECK: FOO1 <> < > <> <> < > <> < > < >
// MINWS: FOO1<><><><><><><><>

TEST(FOO2,)
// CHECK: FOO2 <> < > <> <> < > <> < > < >
// MINWS-SAME: FOO2<><><><><><><><>

TEST(FOO3,)
// CHECK: FOO3 <> < > <> <> < > <> < > < >
// MINWS-SAME: FOO3<><><><><><><><>

TEST(FOO4,)
// CHECK: FOO4 < > < > < > < > < > < > < > < >
// MINWS-SAME: FOO4<><><><><><><><>

TEST(FOO5,)
// CHECK: FOO5 < > < > < > < > < > < > < > < >
// MINWS-SAME: FOO5<><><><><><><><>

TEST(FOO6,)
// CHECK: FOO6 <[]> < []> <[]> <[]> <[] > <[]> <[] > < []>
// MINWS-SAME: FOO6<[]><[]><[]><[]><[]><[]><[]><[]>

TEST(FOO7,)
// CHECK: FOO7 <[ ]> < [ ]> <[ ]> <[ ]> <[ ] > <[ ]> <[ ] > < [ ]>
// MINWS-SAME: FOO7<[]><[]><[]><[]><[]><[]><[]><[]>

TEST(FOO8,)
// CHECK: FOO8 <[ ]> < [ ]> <[ ]> <[ ]> <[ ] > <[ ]> <[ ] > < [ ]>
// MINWS-SAME: FOO8<[]><[]><[]><[]><[]><[]><[]><[]>
