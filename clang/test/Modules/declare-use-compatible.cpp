// Used module not built with -decluse.
// RUN: rm -rf %t
// RUN: %clang_cc1 -x c++ -fmodules -fimplicit-module-maps -fmodule-name=XB -emit-module \
// RUN:   -I %S/Inputs/declare-use %S/Inputs/declare-use/module.modulemap -o %t/b.pcm
// RUN: %clang_cc1 -x c++ -fmodules -fimplicit-module-maps -fmodules-cache-path=%t \
// RUN:   -fmodules-decluse \
// RUN:   -fmodule-file=%t/b.pcm -fmodule-name=XE -I %S/Inputs/declare-use %s
//
// Main module not built with -decluse.
// RUN: rm -rf %t
// RUN: %clang_cc1 -x c++ -fmodules -fimplicit-module-maps -fmodule-name=XB -emit-module \
// RUN:   -fmodules-decluse \
// RUN:   -I %S/Inputs/declare-use %S/Inputs/declare-use/module.modulemap -o %t/b.pcm
// RUN: %clang_cc1 -x c++ -fmodules -fimplicit-module-maps -fmodules-cache-path=%t \
// RUN:   -fmodule-file=%t/b.pcm -fmodule-name=XE -I %S/Inputs/declare-use %s
//
// Used module not built with -decluse.
// RUN: rm -rf %t
// RUN: %clang_cc1 -x c++ -fmodules -fimplicit-module-maps -fmodule-name=XB -emit-module \
// RUN:   -I %S/Inputs/declare-use %S/Inputs/declare-use/module.modulemap -o %t/b.pcm
// RUN: %clang_cc1 -x c++ -fmodules -fimplicit-module-maps -fmodules-cache-path=%t \
// RUN:   -fmodules-strict-decluse \
// RUN:   -fmodule-file=%t/b.pcm -fmodule-name=XE -I %S/Inputs/declare-use %s
//
// Main module not built with -decluse.
// RUN: rm -rf %t
// RUN: %clang_cc1 -x c++ -fmodules -fimplicit-module-maps -fmodule-name=XB -emit-module \
// RUN:   -fmodules-strict-decluse \
// RUN:   -I %S/Inputs/declare-use %S/Inputs/declare-use/module.modulemap -o %t/b.pcm
// RUN: %clang_cc1 -x c++ -fmodules -fimplicit-module-maps -fmodules-cache-path=%t \
// RUN:   -fmodule-file=%t/b.pcm -fmodule-name=XE -I %S/Inputs/declare-use %s

#include "b.h"

const int g = b;

