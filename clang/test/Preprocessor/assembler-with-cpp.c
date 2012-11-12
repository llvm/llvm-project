// RUN: %clang_cc1 -x assembler-with-cpp -E %s -o - | FileCheck -strict-whitespace -check-prefix=CHECK-Identifiers-False %s

#ifndef __ASSEMBLER__
#error "__ASSEMBLER__ not defined"
#endif


// Invalid token pasting is ok. 
#define A X ## .
1: A
// CHECK-Identifiers-False: 1: X .

// Line markers are not linemarkers in .S files, they are passed through.
# 321
// CHECK-Identifiers-False: # 321

// Unknown directives are passed through.
# B C
// CHECK-Identifiers-False: # B C

// Unknown directives are expanded.
#define D(x) BAR ## x
# D(42)
// CHECK-Identifiers-False: # BAR42

// Unmatched quotes are permitted.
2: '
3: "
// CHECK-Identifiers-False: 2: '
// CHECK-Identifiers-False: 3: "

// (balance quotes to keep editors happy): "'

// Empty char literals are ok.
4: ''
// CHECK-Identifiers-False: 4: ''


// Portions of invalid pasting should still expand as macros.
// rdar://6709206
#define M4 expanded
#define M5() M4 ## (

5: M5()
// CHECK-Identifiers-False: 5: expanded (

// rdar://6804322
#define FOO(name)  name ## $foo
6: FOO(blarg)
// CHECK-Identifiers-False: 6: blarg $foo

// RUN: %clang_cc1 -x assembler-with-cpp -fdollars-in-identifiers -E %s -o - | FileCheck -check-prefix=CHECK-Identifiers-True -strict-whitespace %s
#define FOO(name)  name ## $foo
7: FOO(blarg)
// CHECK-Identifiers-True: 7: blarg$foo

// 
#define T6() T6 #nostring
#define T7(x) T7 #x
8: T6()
9: T7(foo)
// CHECK-Identifiers-True: 8: T6 #nostring
// CHECK-Identifiers-True: 9: T7 "foo"

// Concatenation with period doesn't leave a space
#define T8(A,B) A ## B
10: T8(.,T8)
// CHECK-Identifiers-True: 10: .T8

// This should not crash.
#define T11(a) #0
11: T11(b)
// CHECK-Identifiers-True: 11: #0


// This should not crash
// rdar://8823139
# ##
// CHECK-Identifiers-False: # ##
