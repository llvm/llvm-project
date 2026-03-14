// RUN: %clang_cc1 -E -std=c++11 %s -o - | FileCheck -strict-whitespace %s


#define y(a) ..a
A: y(.)
// This should print as ".. ." to avoid turning into ...
// CHECK: A: .. .

#define X 0 .. 1
B: X
// CHECK: B: 0 .. 1

#define DOT .
C: ..DOT
// CHECK: C: .. .


#define PLUS +
#define EMPTY
#define f(x) =x=
D: +PLUS -EMPTY- PLUS+ f(=)
// CHECK: D: + + - - + + = = =


#define test(x) L#x
E: test(str)
// Should expand to L "str" not L"str"
// CHECK: E: L "str"

// Should avoid producing >>=.
#define equal =
F: >>equal
// CHECK: F: >> =

// Make sure we don't introduce spaces in the guid because we try to avoid
// pasting '-' to a numeric constant.
#define TYPEDEF(guid)   typedef [uuid(guid)]
TYPEDEF(66504301-BE0F-101A-8BBB-00AA00300CAB) long OLE_COLOR;
// CHECK: typedef [uuid(66504301-BE0F-101A-8BBB-00AA00300CAB)] long OLE_COLOR;

// Be careful with UD-suffixes.
#define StrSuffix() "abc"_suffix
#define IntSuffix() 123_suffix
UD: StrSuffix()ident
UD: IntSuffix()ident
// CHECK: UD: "abc"_suffix ident
// CHECK: UD: 123_suffix ident
