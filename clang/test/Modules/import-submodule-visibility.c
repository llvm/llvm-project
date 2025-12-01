// This test checks that imports of headers that appeared in a different submodule than
// what is imported by the current TU don't affect the compilation.

// RUN: rm -rf %t
// RUN: split-file %s %t

//--- C/C.h
#include "Textual.h"
//--- C/module.modulemap
module C { header "C.h" }

//--- D/D1.h
#include "Textual.h"
//--- D/D2.h
//--- D/module.modulemap
module D {
  module D1 { header "D1.h" }
  module D2 { header "D2.h" }
}

//--- E/E1.h
#include "E2.h"
//--- E/E2.h
#include "Textual.h"
//--- E/module.modulemap
module E {
  module E1 { header "E1.h" }
  module E2 { header "E2.h" }
}

//--- Textual.h
#define MACRO_TEXTUAL 1

//--- test_top.c
#import "Textual.h"
static int x = MACRO_TEXTUAL;

//--- test_sub.c
#import "D/D2.h"
#import "Textual.h"
static int x = MACRO_TEXTUAL;

//--- test_transitive.c
#import "E/E1.h"
#import "Textual.h"
static int x = MACRO_TEXTUAL;

// Simply loading a PCM file containing top-level module including a header does
// not prevent inclusion of that header in the TU.
//
// RUN: %clang_cc1 -fmodules -I %t -emit-module %t/C/module.modulemap -fmodule-name=C -o %t/C.pcm
// RUN: %clang_cc1 -fmodules -I %t -fsyntax-only %t/test_top.c -fmodule-file=%t/C.pcm

// Loading a PCM file and importing its empty submodule does not prevent
// inclusion of headers included by invisible sibling submodules.
//
// RUN: %clang_cc1 -fmodules -I %t -emit-module %t/D/module.modulemap -fmodule-name=D -o %t/D.pcm
// RUN: %clang_cc1 -fmodules -I %t -fsyntax-only %t/test_sub.c -fmodule-file=%t/D.pcm

// Loading a PCM file and importing a submodule does not prevent inclusion of
// headers included by some of its transitive un-exported dependencies.
//
// RUN: %clang_cc1 -fmodules -I %t -emit-module %t/E/module.modulemap -fmodule-name=E -o %t/E.pcm
// RUN: %clang_cc1 -fmodules -I %t -fsyntax-only %t/test_transitive.c -fmodule-file=%t/E.pcm
