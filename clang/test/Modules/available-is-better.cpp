// RUN: rm -rf %t
// RUN: %clang_cc1 -x c++ -Rmodule-build -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs/available-is-better %s 2>&1 | FileCheck %s

#include "available-is-better.h"
// CHECK: remark: building module
