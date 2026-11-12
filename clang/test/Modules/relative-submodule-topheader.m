// RUN: cd %S
// RUN: %clang_cc1 -fmodules -fno-implicit-modules -x objective-c++ -fmodule-name=std -emit-module Inputs/submodules/module.modulemap -o %t/mod.pcm
// RUN: llvm-bcanalyzer --dump --disable-histogram %t/mod.pcm | FileCheck %s

// CHECK: <SUBMODULE_HEADER abbrevid=6/> blob data = 'vector.h'
// CHECK: <SUBMODULE_TOPHEADER abbrevid=7/> blob data = 'vector.h'
// CHECK: <SUBMODULE_HEADER abbrevid=6/> blob data = 'type_traits.h'
// CHECK: <SUBMODULE_TOPHEADER abbrevid=7/> blob data = 'type_traits.h'
// CHECK: <SUBMODULE_HEADER abbrevid=6/> blob data = 'hash_map.h'
// CHECK: <SUBMODULE_TOPHEADER abbrevid=7/> blob data = 'hash_map.h'
