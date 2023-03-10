// RUN: rm -rf %t
// RUN: %clang_cc1 -std=c++11 -dwarf-ext-refs -fmodule-format=obj \
// RUN:     -fmodule-map-file=%S/Inputs/gmodules-preferred-name-alias.modulemap \
// RUN:     -fmodules-cache-path=%t -debug-info-kind=standalone -debugger-tuning=lldb \
// RUN:     -fmodules -mllvm -debug-only=pchcontainer -x c++ \
// RUN:     -I %S/Inputs %s &> %t.ll
// RUN: cat %t.ll | FileCheck %s

#include "gmodules-preferred-name-alias.h"

// CHECK: ![[#]] = !DIDerivedType(tag: DW_TAG_typedef, name: "Bar<char>", scope: ![[#]], file: ![[#]], line: [[#]], baseType: ![[PREF_BASE:[0-9]+]])
// CHECK: ![[PREF_BASE]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo<char>"
