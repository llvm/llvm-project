/*
RUN: %clang_cc1 %s -std=gnu89 -fsyntax-only 2>&1 | FileCheck -check-prefix=CHECK-GNU89 %s -allow-empty
RUN: %clang_cc1 %s -std=gnu89 -pedantic -fsyntax-only 2>&1 | FileCheck -check-prefix=CHECK-GNU89-PEDANTIC %s
*/

typedef const int t;
const t c_i;
/*
CHECK-GNU89-NOT: 7:1: warning: duplicate 'const' declaration specifier
CHECK-GNU89-PEDANTIC: 7:1: warning: duplicate 'const' declaration specifier
*/

const int c_i2;
const typeof(c_i2) c_i3;
/*
CHECK-GNU89-NOT: 14:7: warning: extension used
CHECK-GNU89-NOT: 14:1: warning: duplicate 'const' declaration specifier
CHECK-GNU89-PEDANTIC: 14:7: warning: extension used
CHECK-GNU89-PEDANTIC: 14:1: warning: duplicate 'const' declaration specifier
*/
