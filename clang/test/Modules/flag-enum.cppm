// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -std=c++11 -x c++ -fmodules -emit-module \
// RUN:   -fmodule-name=ClangModule %t/module.modulemap -o %t/ClangModule.pcm \
// RUN:   -verify=module
// RUN: %clang_cc1 -std=c++11 -fmodules -fno-implicit-modules \
// RUN:   -fmodule-map-file=%t/module.modulemap \
// RUN:   -fmodule-file=ClangModule=%t/ClangModule.pcm -fsyntax-only \
// RUN:   -verify=use %t/use-clang-module.cpp
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/NamedModule.cppm \
// RUN:   -o %t/NamedModule.pcm -verify=module
// RUN: %clang_cc1 -std=c++20 -fmodule-file=NamedModule=%t/NamedModule.pcm \
// RUN:   -fsyntax-only -verify=use %t/use-named-module.cpp

//--- module.modulemap
module ClangModule {
  header "clang-module.h"
}

//--- clang-module.h
enum class [[clang::flag_enum]] ClangModuleFlags { A = 1, B = 2 };
// module-warning@-1 {{'operator|' is not available for flag-like enumeration type 'ClangModuleFlags'}}
// module-warning@-2 {{'operator&' is not available for flag-like enumeration type 'ClangModuleFlags'}}
// module-warning@-3 {{'operator^' is not available for flag-like enumeration type 'ClangModuleFlags'}}
// module-warning@-4 {{'operator~' is not available for flag-like enumeration type 'ClangModuleFlags'}}

//--- use-clang-module.cpp
// use-no-diagnostics
#include "clang-module.h"

//--- NamedModule.cppm
export module NamedModule;
export enum class [[clang::flag_enum]] NamedModuleFlags { A = 1, B = 2 };
// module-warning@-1 {{'operator|' is not available for flag-like enumeration type 'NamedModuleFlags'}}
// module-warning@-2 {{'operator&' is not available for flag-like enumeration type 'NamedModuleFlags'}}
// module-warning@-3 {{'operator^' is not available for flag-like enumeration type 'NamedModuleFlags'}}
// module-warning@-4 {{'operator~' is not available for flag-like enumeration type 'NamedModuleFlags'}}

//--- use-named-module.cpp
// use-no-diagnostics
import NamedModule;
NamedModuleFlags flags;
