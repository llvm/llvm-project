// RUN: cd %S
// RUN: rm -rf %t
// RUN: mkdir %t
//
// RUN: %clang_cc1 -fmodules -fno-implicit-modules -fmodule-name=relative-dep-gen -emit-module -x c++ Inputs/relative-dep-gen.modulemap -dependency-file %t/build.d -MT mod.pcm -o %t/mod.pcm
// RUN: %clang_cc1 -fmodules -fno-implicit-modules -fmodule-map-file=Inputs/relative-dep-gen.modulemap -fmodule-file=%t/mod.pcm -dependency-file %t/use-explicit.d -MT use.o relative-dep-gen.cpp -fsyntax-only
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -fmodule-map-file=Inputs/relative-dep-gen.modulemap -dependency-file %t/use-implicit.d relative-dep-gen.cpp -MT use.o -fsyntax-only
//
// RUN: FileCheck --check-prefix=CHECK-BUILD %s < %t/build.d
// RUN: FileCheck --check-prefix=CHECK-USE --check-prefix=CHECK-EXPLICIT %s < %t/use-explicit.d
// RUN: FileCheck --check-prefix=CHECK-USE --check-prefix=CHECK-IMPLICIT %s < %t/use-implicit.d
//
// RUN: %clang_cc1 -fmodules -fno-implicit-modules -fmodule-name=relative-dep-gen -emit-module -x c++ Inputs/relative-dep-gen-cwd.modulemap -dependency-file %t/build-cwd.d -MT mod.pcm -o %t/mod-cwd.pcm -fmodule-map-file-home-is-cwd
// RUN: %clang_cc1 -fmodules -fno-implicit-modules -fmodule-map-file=Inputs/relative-dep-gen-cwd.modulemap -fmodule-file=%t/mod-cwd.pcm -dependency-file %t/use-explicit-cwd.d -MT use.o relative-dep-gen.cpp -fsyntax-only -fmodule-map-file-home-is-cwd
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -fmodule-map-file=Inputs/relative-dep-gen-cwd.modulemap -dependency-file %t/use-implicit-cwd.d relative-dep-gen.cpp -MT use.o -fsyntax-only -fmodule-map-file-home-is-cwd
//
// RUN: FileCheck --check-prefix=CHECK-BUILD %s < %t/build-cwd.d
// RUN: FileCheck --check-prefix=CHECK-USE %s < %t/use-explicit-cwd.d
// RUN: FileCheck --check-prefix=CHECK-USE %s < %t/use-implicit-cwd.d
//
// Check that the .d file is still correct after relocating the module.
// RUN: mkdir %t/Inputs
// RUN: cp %S/Inputs/relative-dep-gen-1.h %t/Inputs
// RUN: cp %s %t
// RUN: cd %t
// RUN: %clang_cc1 -fmodules -fno-implicit-modules -fmodule-file=%t/mod-cwd.pcm -dependency-file %t/use-explicit-no-map-cwd.d -MT use.o relative-dep-gen.cpp -fsyntax-only -fmodule-map-file-home-is-cwd
// RUN: cat %t/use-explicit-no-map-cwd.d
// RUN: FileCheck --check-prefix=CHECK-USE %s < %t/use-explicit-no-map-cwd.d

#include "Inputs/relative-dep-gen-1.h"

// CHECK-BUILD: mod.pcm:
// CHECK-BUILD:   {{[ \t]}}Inputs/relative-dep-gen{{(-cwd)?}}.modulemap
// CHECK-BUILD:   {{[ \t]}}Inputs/relative-dep-gen-1.h
// CHECK-BUILD:   {{[ \t]}}Inputs/relative-dep-gen-2.h
// CHECK-USE: use.o:
// CHECK-USE-DAG:   {{[ \t]}}relative-dep-gen.cpp
// CHECK-EXPLICIT-DAG:   mod.pcm
// CHECK-IMPLICIT-DAG:   {{[ \t]}}Inputs{{[/\\]}}relative-dep-gen-1.h
