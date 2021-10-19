// RUN: rm -rf %t && mkdir %t
//
// RUN: %clang_cc1 -fmodules -fno-implicit-modules -fimplicit-module-maps \
// RUN:   -emit-module -x c++ %S/Inputs/explicit-build-inferred/frameworks/module.modulemap \
// RUN:   -fmodule-name=Inferred -o %t/Inferred.pcm -F %S/Inputs/explicit-build-inferred/frameworks
//
// RUN: %clang_cc1 -fmodules -fno-implicit-modules -fsyntax-only %s \
// RUN:   -fmodule-map-file=%S/Inputs/explicit-build-inferred/frameworks/module.modulemap \
// RUN:   -fmodule-file=%t/Inferred.pcm -F %S/Inputs/explicit-build-inferred/frameworks

#include <Inferred/Inferred.h>

inferred a = 0;
