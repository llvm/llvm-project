// RUN: %clang_cc1 -fsyntax-only -Wno-unused-value -fno-diagnostics-show-line-numbers %s 2>&1 | FileCheck %s
// REQUIRES: asserts

void foo(void) {

  "§Ã"; // ø
// CHECK: {{^  "<A7><C3>"; // <F8>}}
// CHECK: {{^   \^~~~~~~}}

  /* þ« */ const char *d = "¥";

// CHECK: {{^  /\* <FE><AB> \*/ const char \*d = "<A5>";}}
// CHECK: {{^                                  \^~~~}}

  "xxé¿¿¿d";
// CHECK: {{^  "xxé¿¿<BF>d";}}
// CHECK: {{^             \^~~~}}

  "xxé¿bcd";
// CHECK: {{^  "xx<E9><BF>bcd";}}
// CHECK: {{^     \^~~~~~~~}}

  "xxéabcd";
// CHECK: {{^  "xx<E9>abcd";}}
// CHECK: {{^     \^~~~}}

  "xxé¿é¿d";
// CHECK: {{^  "xx<E9><BF><E9><BF>d";}}
// CHECK: {{^     \^~~~~~~~~~~~~~~}}

  "xxé¿xxxxxxxxxxxxxxxxxxxxxé¿xx";
// CHECK: {{^  "xx<E9><BF>xxxxxxxxxxxxxxxxxxxxx<E9><BF>xx";}}
// CHECK: {{^     \^~~~~~~~                     ~~~~~~~~}}

  "?kÍ›S¥ÇØg7†,	2,Díu„†*É,pûäÚ&”‰(K§:Ñ'1á‹ÎjOÅ°<:";

  "xé¿xé¿xé¿xé¿xé¿xé¿xé¿xé¿xé¿xé¿xé¿xé¿x";
}
// CHECK-NOT:Assertion
