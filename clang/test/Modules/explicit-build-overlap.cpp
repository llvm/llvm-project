// RUN: rm -rf %t
// RUN: %clang_cc1 -x c++ -fmodules                   %S/Inputs/explicit-build-overlap/def.map -fmodule-name=a -emit-module -o %t/a.pcm
// RUN: %clang_cc1 -x c++ -fmodules                   %S/Inputs/explicit-build-overlap/def.map -fmodule-name=b -emit-module -o %t/ba.pcm -fmodule-file=%t/a.pcm
// RUN: %clang_cc1 -x c++ -fmodules -fmodule-map-file=%S/Inputs/explicit-build-overlap/use.map -fmodule-name=use -fmodule-file=%t/ba.pcm %s -verify -I%S/Inputs/explicit-build-overlap -fmodules-decluse
//
// RUN: %clang_cc1 -x c++ -fmodules                   %S/Inputs/explicit-build-overlap/def.map -fmodule-name=b -emit-module -o %t/b.pcm
// RUN: %clang_cc1 -x c++ -fmodules                   %S/Inputs/explicit-build-overlap/def.map -fmodule-name=a -emit-module -o %t/ab.pcm -fmodule-file=%t/b.pcm
// RUN: %clang_cc1 -x c++ -fmodules -fmodule-map-file=%S/Inputs/explicit-build-overlap/use.map -fmodule-name=use -fmodule-file=%t/ab.pcm %s -verify -I%S/Inputs/explicit-build-overlap -fmodules-decluse

// expected-no-diagnostics
#include "a.h"

A a;
B b;
