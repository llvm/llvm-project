// RUN: rm -rf %t
// RUN: mkdir %t

// RUN: echo '41:c:@S@G@F@G#@Sa@F@operator void (*)(int)#1 %/t/importee.ast' >> %t/externalDefMap.txt
// RUN: echo '38:c:@S@G@F@G#@Sa@F@operator void (*)()#1 %/t/importee.ast' >> %t/externalDefMap.txt
// RUN: echo '14:c:@F@importee# %/t/importee.ast' >> %t/externalDefMap.txt
// RUN: %clang_cc1 -emit-pch %/S/Inputs/ctu-lookup-name-with-space.cpp -o %t/importee.ast

// RUN: cd %t
// RUN: %clang_cc1 -fsyntax-only -analyze \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=. \
// RUN:   -analyzer-config display-ctu-progress=true \
// RUN:   -verify %s 2>&1 | FileCheck %s

// CHECK: CTU loaded AST file

// FIXME: In this test case, we cannot use the on-demand-parsing approach to
//        load the external TU.
//
//        In the Darwin system, the target triple is determined by the driver,
//        rather than using the default one like other systems. However, when
//        using bare `clang -cc1`, the adjustment is not done, which cannot
//        match the one loaded with on-demand-parsing (adjusted triple).
//        We bypass this problem by loading AST files, whose target triple is
//        also unadjusted when generated via `clang -cc1 -emit-pch`.
//
//        Refer to: https://discourse.llvm.org/t/60762
//
//        This is also the reason why the test case of D75665 (introducing
//        the on-demand-parsing feature) is enabled only on Linux.

void importee();

void trigger() {
  // Call an external function to trigger the parsing process of CTU index.
  // Refer to file Inputs/ctu-lookup-name-with-space.cpp for more details.

  importee(); // expected-no-diagnostics
}
