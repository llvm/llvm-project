// Test with pch.
// RUN: %clang_cc1 -emit-pch -DFOO -o %t %S/variables.h
// RUN: %clang_cc1 -DBAR=int -include-pch %t -fsyntax-only -pedantic %s
// RUN: %clang_cc1 -DFOO -DBAR=int -include-pch %t %s
// RUN: not %clang_cc1 -DFOO=blah -DBAR=int -include-pch %t %s 2> %t.err
// RUN: FileCheck -check-prefix=CHECK-FOO %s < %t.err
// RUN: not %clang_cc1 -UFOO -include-pch %t %s 2> %t.err
// RUN: FileCheck -check-prefix=CHECK-NOFOO %s < %t.err

// RUN: not %clang_cc1 -DFOO -undef -include-pch %t %s 2> %t.err
// RUN: FileCheck -check-prefix=CHECK-UNDEF %s < %t.err

BAR bar = 17;

#ifndef FOO
#  error FOO was not defined
#endif

#if FOO != 1
#  error FOO has the wrong definition
#endif

#ifndef BAR
#  error BAR was not defined
#endif

// CHECK-FOO: definition of macro 'FOO' differs between the AST file '{{.*}}' ('1') and the command line ('blah')
// CHECK-NOFOO: macro 'FOO' was defined in the AST file '{{.*}}' but undef'd on the command line

// CHECK-UNDEF: command line contains '-undef' but AST file '{{.*}}' was not built with it

