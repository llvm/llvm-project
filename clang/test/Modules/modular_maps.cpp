// RUN: rm -rf %t
//
// RUN: %clang_cc1 -x objective-c++ -fmodules-cache-path=%t -fmodules -fmodule-map-file=%S/Inputs/modular_maps/modulea.map -fmodule-map-file=%S/Inputs/modular_maps/modulec.map -I %S/Inputs/modular_maps %s -verify
// RUN: %clang_cc1 -x objective-c++ -fmodules-cache-path=%t -fmodules -fmodule-map-file=%S/Inputs/modular_maps/modulec.map -fmodule-map-file=%S/Inputs/modular_maps/modulea.map -I %S/Inputs/modular_maps %s -verify
//
// RUN: cd %S
// RUN: %clang_cc1 -x objective-c++ -fmodules-cache-path=%t -fmodules -fmodule-map-file=Inputs/modular_maps/modulea.map -fmodule-map-file=Inputs/modular_maps/modulec.map -I Inputs/modular_maps %s -verify
// RUN: %clang_cc1 -x objective-c++ -fmodules-cache-path=%t -fmodules -fmodule-map-file=Inputs/modular_maps/modulec.map -fmodule-map-file=Inputs/modular_maps/modulea.map -I Inputs/modular_maps %s -verify
//
// RUN: cd %S
// RUN: %clang_cc1 -x objective-c++ -fmodules-cache-path=%t -fmodules -fmodule-map-file=Inputs/modular_maps/modulea-cwd.map -fmodule-map-file=Inputs/modular_maps/modulec-cwd.map -I Inputs/modular_maps %s -verify -fmodule-map-file-home-is-cwd
// RUN: %clang_cc1 -x objective-c++ -fmodules-cache-path=%t -fmodules -fmodule-map-file=Inputs/modular_maps/modulec-cwd.map -fmodule-map-file=Inputs/modular_maps/modulea-cwd.map -I Inputs/modular_maps %s -verify -fmodule-map-file-home-is-cwd

#include "common.h"
#include "a.h"
#include "b.h" // expected-error {{private header}}
@import C;
const int v = a + c + x;
const int val = a + b + c + x; // expected-error {{undeclared identifier}}
