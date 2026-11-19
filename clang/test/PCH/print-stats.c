// Declarations and types of a PCH that aren't used must not be counted as
// read, even if they are in pages of the vectors that haven't been
// materialized.

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -include-pch %t \
// RUN:   -fsyntax-only -print-stats %s 2>&1 | FileCheck %s

// CHECK: *** AST File Statistics:
// CHECK: {{^  [0-9]{1,2}/[0-9]{4} declarations read}}

#ifndef HEADER
#define HEADER

#define D1(x) void f##x(void);
#define D10(x) D1(x##0) D1(x##1) D1(x##2) D1(x##3) D1(x##4) \
               D1(x##5) D1(x##6) D1(x##7) D1(x##8) D1(x##9)
#define D100(x) D10(x##0) D10(x##1) D10(x##2) D10(x##3) D10(x##4) \
                D10(x##5) D10(x##6) D10(x##7) D10(x##8) D10(x##9)
#define D1000(x) D100(x##0) D100(x##1) D100(x##2) D100(x##3) D100(x##4) \
                 D100(x##5) D100(x##6) D100(x##7) D100(x##8) D100(x##9)

D1000(1)
D1000(2)

#else

void use(void);

#endif
