//NOTE: this test reuses the existing include/module files.

// RUN: rm -rf %t
// RUN: mkdir -p %t/merge-vtable-codegen
// RUN: cp -r %S/Inputs/merge-vtable-codegen %t/

// RUN: %clang_cc1 -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodule-name=b -fmodules-cache-path=%t -o %t/b1.pcm \
// RUN:     -emit-module %t/merge-vtable-codegen/merge-vtable-codegen.modulemap \
// RUN:     -I%t//merge-vtable-codegen -fmacro-prefix-map=%S=x:/PREFIX-MAP-IN -fmacro-prefix-map=%t=x:/PREFIX-MAP-OUT
// RUN: llvm-bcanalyzer -dump --disable-histogram %t/b1.pcm | FileCheck -check-prefix=CHECK-PREFIX-MAP-OUT %s 

// RUN: %clang_cc1 -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodule-name=b -fmodules-cache-path=%t -o %t/b1.pcm \
// RUN:     -emit-module %t/merge-vtable-codegen/merge-vtable-codegen.modulemap \
// RUN:     -I%t/./merge-vtable-codegen -fmacro-prefix-map=%S=x:/PREFIX-MAP-IN -fmacro-prefix-map=%t=x:/PREFIX-MAP-OUT
// RUN: llvm-bcanalyzer -dump --disable-histogram %t/b1.pcm | FileCheck -check-prefix=CHECK-PREFIX-MAP-OUT %s 

// RUN: %clang_cc1 -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodule-name=b -fmodules-cache-path=%t -o %t/b1.pcm \
// RUN:     -emit-module %t/merge-vtable-codegen/merge-vtable-codegen.modulemap \
// RUN:     -I%t/../merge-vtable-codegen -fmacro-prefix-map=%S=x:/PREFIX-MAP-IN -fmacro-prefix-map=%t=x:/PREFIX-MAP-OUT
// RUN: llvm-bcanalyzer -dump --disable-histogram %t/b1.pcm | FileCheck -check-prefix=CHECK-PREFIX-MAP-OUT %s 


// First, build two modules that both re-export the same header.
// RUN: %clang_cc1 -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodule-name=b -o %t/b.pcm \
// RUN:     -emit-module %S/Inputs/merge-vtable-codegen/merge-vtable-codegen.modulemap \
// RUN:     -I%S/Inputs/merge-vtable-codegen -fmacro-prefix-map=%S=x:/PREFIX-MAP-IN -fmacro-prefix-map=%t=x:/PREFIX-MAP-OUT
// RUN: llvm-bcanalyzer -dump --disable-histogram %t/b.pcm | FileCheck -check-prefix=CHECK-PREFIX-MAP-IN %s 

// RUN: %clang_cc1 -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodule-name=c -o %t/c.pcm \
// RUN:     -emit-module %S/Inputs/merge-vtable-codegen/merge-vtable-codegen.modulemap \
// RUN:     -I%S/Inputs/merge-vtable-codegen -fmacro-prefix-map=%S=x:/PREFIX-MAP-IN -fmacro-prefix-map=%t=x:/PREFIX-MAP-OUT
// RUN: llvm-bcanalyzer -dump --disable-histogram %t/c.pcm | FileCheck -check-prefix=CHECK-PREFIX-MAP-IN %s 

// Use the two modules in a single compile.
// RUN: %clang_cc1 -x c++ -std=c++11 -fmodules -fimplicit-module-maps -fmodule-file=%t/b.pcm -fmodule-file=%t/c.pcm \
// RUN:     -fmodule-map-file=%S/Inputs/merge-vtable-codegen/merge-vtable-codegen.modulemap  -fmacro-prefix-map=%S=x:/PREFIX-MAP-IN -fmacro-prefix-map=%t=x:/PREFIX-MAP-OUT \
// RUN:     -emit-llvm -o %t/test.o %s


// CHECK-PREFIX-MAP-IN: <MODULE_DIRECTORY {{.*}}/> blob data = 'x:/PREFIX-MAP-IN{{[/\\]}}Inputs{{[/\\]}}merge-vtable-codegen'

// CHECK-PREFIX-MAP-OUT: <MODULE_DIRECTORY {{.*}}/> blob data = 'x:/PREFIX-MAP-OUT{{[/\\]}}merge-vtable-codegen'

#include "Inputs/merge-vtable-codegen/c.h"
#include "Inputs/merge-vtable-codegen/b.h"
