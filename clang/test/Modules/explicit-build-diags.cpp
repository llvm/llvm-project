// RUN: rm -rf %t && mkdir %t
// RUN: %clang_cc1 -fmodules -x c++ %S/Inputs/explicit-build-diags/module.modulemap -fmodule-name=a -emit-module -o %t/a.pcm
// RUN: %clang_cc1 -fmodules -Wdeprecated-declarations -fdiagnostics-show-note-include-stack -serialize-diagnostic-file %t/tu.dia \
// RUN:   -I %S/Inputs/explicit-build-diags -fmodule-file=%t/a.pcm -fsyntax-only %s

#include "a.h"

void foo() { a(); }
