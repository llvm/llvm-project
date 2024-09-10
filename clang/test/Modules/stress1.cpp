// RUN: rm -rf %t
// RUN: cd %S
//
// RUN: %clang_cc1 -fmodules -x c++ -std=c++11 \
// RUN:   -I Inputs/stress1 \
// RUN:   -fno-implicit-modules \
// RUN:   -fmodules-cache-path=%t \
// RUN:   -fmodule-map-file-home-is-cwd \
// RUN:   -emit-module -fmodule-name=m00 -o %t/m00.pcm \
// RUN:   Inputs/stress1/module.modulemap
//
// RUN: %clang_cc1 -fmodules -x c++ -std=c++11 \
// RUN:   -I Inputs/stress1 \
// RUN:   -fno-implicit-modules \
// RUN:   -fmodules-cache-path=%t \
// RUN:   -fmodule-map-file-home-is-cwd \
// RUN:   -emit-module -fmodule-name=m00 -o %t/m00_check.pcm \
// RUN:   Inputs/stress1/module.modulemap
//
// RUN: diff %t/m00.pcm %t/m00_check.pcm
//
// RUN: %clang_cc1 -fmodules -x c++ -std=c++11 -fdelayed-template-parsing \
// RUN:   -I Inputs/stress1 \
// RUN:   -fno-implicit-modules \
// RUN:   -fmodules-cache-path=%t \
// RUN:   -fmodule-map-file-home-is-cwd \
// RUN:   -emit-module -fmodule-name=m01 -o %t/m01.pcm \
// RUN:   Inputs/stress1/module.modulemap
//
// RUN: %clang_cc1 -fmodules -x c++ -std=c++11 -fdelayed-template-parsing \
// RUN:   -I Inputs/stress1 \
// RUN:   -fno-implicit-modules \
// RUN:   -fmodules-cache-path=%t \
// RUN:   -fmodule-map-file-home-is-cwd \
// RUN:   -emit-module -fmodule-name=m01 -o %t/m01_check.pcm \
// RUN:   Inputs/stress1/module.modulemap
//
// RUN: diff %t/m01.pcm %t/m01_check.pcm
//
// RUN: %clang_cc1 -fmodules -x c++ -std=c++11 \
// RUN:   -I Inputs/stress1 \
// RUN:   -fno-implicit-modules \
// RUN:   -fmodules-cache-path=%t \
// RUN:   -fmodule-map-file-home-is-cwd \
// RUN:   -emit-module -fmodule-name=m02 -o %t/m02.pcm \
// RUN:   Inputs/stress1/module.modulemap
//
// RUN: %clang_cc1 -fmodules -x c++ -std=c++11 \
// RUN:   -I Inputs/stress1 \
// RUN:   -fno-implicit-modules \
// RUN:   -fmodules-cache-path=%t \
// RUN:   -fmodule-map-file-home-is-cwd \
// RUN:   -emit-module -fmodule-name=m03 -o %t/m03.pcm \
// RUN:   Inputs/stress1/module.modulemap
//
// RUN: %clang_cc1 -fmodules -x c++ -std=c++11 \
// RUN:   -I Inputs/stress1 \
// RUN:   -fno-implicit-modules \
// RUN:   -fmodules-cache-path=%t \
// RUN:   -fmodule-map-file-home-is-cwd \
// RUN:   -fmodule-file=%t/m00.pcm \
// RUN:   -fmodule-file=%t/m01.pcm \
// RUN:   -fmodule-file=%t/m02.pcm \
// RUN:   -fmodule-file=%t/m03.pcm \
// RUN:   -emit-module -fmodule-name=merge00 -o %t/merge00.pcm \
// RUN:   Inputs/stress1/module.modulemap
//
// RUN: %clang_cc1 -fmodules -x c++ -std=c++11 \
// RUN:   -I Inputs/stress1 \
// RUN:   -fno-implicit-modules \
// RUN:   -fmodules-cache-path=%t \
// RUN:   -fmodule-map-file-home-is-cwd \
// RUN:   -fmodule-file=%t/m00.pcm \
// RUN:   -fmodule-file=%t/m01.pcm \
// RUN:   -fmodule-file=%t/m02.pcm \
// RUN:   -fmodule-file=%t/m03.pcm \
// RUN:   -emit-module -fmodule-name=merge00 -o %t/merge00_check.pcm \
// RUN:   Inputs/stress1/module.modulemap
//
// RUN: diff %t/merge00.pcm %t/merge00_check.pcm
//
// RUN: %clang_cc1 -fmodules -x c++ -std=c++11 \
// RUN:   -I Inputs/stress1 \
// RUN:   -fno-implicit-modules \
// RUN:   -fmodules-cache-path=%t \
// RUN:   -fmodule-map-file-home-is-cwd \
// RUN:   -fmodule-map-file=Inputs/stress1/module.modulemap \
// RUN:   -fmodule-file=%t/m00.pcm \
// RUN:   -fmodule-file=%t/m01.pcm \
// RUN:   -fmodule-file=%t/m02.pcm \
// RUN:   -fmodule-file=%t/m03.pcm \
// RUN:   -fmodule-file=%t/merge00.pcm \
// RUN:   -verify stress1.cpp -emit-llvm -o %t/stress1.ll
//
// RUN: %clang_cc1 -fmodules -x c++ -std=c++11 \
// RUN:   -I Inputs/stress1 \
// RUN:   -fno-implicit-modules \
// RUN:   -fmodules-cache-path=%t \
// RUN:   -fmodule-map-file-home-is-cwd \
// RUN:   -fmodule-map-file=Inputs/stress1/module.modulemap \
// RUN:   -fmodule-file=%t/m00.pcm \
// RUN:   -fmodule-file=%t/m01.pcm \
// RUN:   -fmodule-file=%t/m02.pcm \
// RUN:   -fmodule-file=%t/m03.pcm \
// RUN:   -fmodule-file=%t/merge00.pcm \
// RUN:   -verify stress1.cpp -emit-llvm -o %t/stress1_check.ll
//
// RUN: diff -u %t/stress1.ll %t/stress1_check.ll
//
// RUN: %clang_cc1 -fmodules -x c++ -std=c++11 \
// RUN:   -I Inputs/stress1 \
// RUN:   -fmodules-cache-path=%t \
// RUN:   -fmodule-map-file-home-is-cwd \
// RUN:   -fmodule-file=%t/m00.pcm \
// RUN:   -fmodule-file=%t/m01.pcm \
// RUN:   -fmodule-file=%t/m02.pcm \
// RUN:   -fmodule-file=%t/m03.pcm \
// RUN:   -emit-module -fmodule-name=merge00 -o /dev/null \
// RUN:   -DMERGE_NO_REEXPORT \
// RUN:   Inputs/stress1/module.modulemap
//
// expected-no-diagnostics

#include "m00.h"
#include "m01.h"
#include "m02.h"
#include "m03.h"

#include "merge00.h"

int f() { return N01::S00('a').method00('b') + (int)N00::S00(42) + function00(42) + g(); }

int f2() {
  return pragma_weak00() + pragma_weak01() + pragma_weak02() +
         pragma_weak03 + pragma_weak04 + pragma_weak05;
}
