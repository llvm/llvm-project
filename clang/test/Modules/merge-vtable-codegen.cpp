// RUN: rm -rf %t

// First, build two modules that both re-export the same header.
// RUN: %clang_cc1 -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodule-name=b -o %t/b.pcm \
// RUN:     -emit-module %S/Inputs/merge-vtable-codegen/merge-vtable-codegen.modulemap \
// RUN:     -I %S/Inputs/merge-vtable-codegen
// RUN: %clang_cc1 -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodule-name=c -o %t/c.pcm \
// RUN:     -emit-module %S/Inputs/merge-vtable-codegen/merge-vtable-codegen.modulemap \
// RUN:     -I %S/Inputs/merge-vtable-codegen

// Use the two modules in a single compile.
// RUN: %clang_cc1 -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodule-file=%t/b.pcm -fmodule-file=%t/c.pcm \
// RUN:     -fmodule-map-file=%S/Inputs/merge-vtable-codegen/merge-vtable-codegen.modulemap \
// RUN:     -emit-llvm -o %t/test.o %s

// Note that order is important:
// Module 'c' just reexports A, while module 'b' defines a method that uses a
// virtual method of A.
#include "Inputs/merge-vtable-codegen/c.h"
#include "Inputs/merge-vtable-codegen/b.h"

void t() {
  b(nullptr);
}
