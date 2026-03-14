// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -x c++ -std=c++11 -fmodules-cache-path=%t -I %S/Inputs/cxx-dtor -emit-llvm-only %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -x c++ -std=c++11 -fmodules-cache-path=%t -I %S/Inputs/cxx-dtor -emit-llvm-only %s -triple i686-windows
#include "b.h"
