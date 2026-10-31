// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -x c++ -std=c++20 %t/A.cppm -I%t -emit-module-interface -o %t/A.pcm
// RUN: %clang_cc1 -x c++ -std=c++20 %t/B.cppm -I%t -emit-module-interface -o %t/B.pcm
// RUN: %clang_cc1 -x c++ -std=c++20 -fprebuilt-module-path=%t %t/use.cpp -verify
//
// RUN: rm %t/A.pcm %t/B.pcm
// RUN: %clang_cc1 -x c++ -std=c++20 %t/A.cppm -I%t -emit-reduced-module-interface -o %t/A.pcm
// RUN: %clang_cc1 -x c++ -std=c++20 %t/B.cppm -I%t -emit-reduced-module-interface -o %t/B.pcm
// RUN: %clang_cc1 -x c++ -std=c++20 -fprebuilt-module-path=%t %t/use.cpp -verify

// Regression test for the StmtProfiler::VisitGCCAsmStmt fix in
// clang/lib/AST/StmtProfile.cpp (using Visit() so StringLiteral bytes are
// folded into the FoldingSetNodeID, and therefore into the ODR hash).
// Without the fix, two inline functions whose only difference is the inline
// asm body produce the same ODR hash, get merged across modules, and no
// diagnostic is emitted at the use site.

//--- a.h
inline int asm_string_func() {
  int x = 0;
  __asm__("foo" : "+r"(x));
  return x;
}
inline int clobber_func() {
  int x = 0;
  __asm__("" : "+r"(x) : : "memory");
  return x;
}

//--- a.v1.h
inline int asm_string_func() {
  int x = 0;
  __asm__("bar" : "+r"(x));        // differs from a.h: asm string
  return x;
}
inline int clobber_func() {
  int x = 0;
  __asm__("" : "+r"(x) : : "cc");  // differs from a.h: clobber
  return x;
}

//--- A.cppm
module;
#include "a.h"
export module A;
export using ::asm_string_func;
export using ::clobber_func;

//--- B.cppm
module;
#include "a.v1.h"
export module B;
export using ::asm_string_func;
export using ::clobber_func;

//--- use.cpp
import A;
import B;
// expected-error@*:* 1+{{'asm_string_func' has different definitions in different modules}}
// expected-error@*:* 1+{{'clobber_func' has different definitions in different modules}}
// expected-note@*:* 1+{{but in 'A.<global>' found a different body}}

int u1 = asm_string_func();
int u2 = clobber_func();
