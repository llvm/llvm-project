// RUN: rm -rf %t

// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t/modules.cache \
// RUN:  -I %S/Inputs/odr_hash-Friend \
// RUN:  -emit-obj -o /dev/null \
// RUN:  -fmodules \
// RUN:  -fimplicit-module-maps \
// RUN:  -fmodules-cache-path=%t/modules.cache \
// RUN:  -std=c++11 -x c++ %s -verify

// UNSUPPORTED: system-windows

// expected-no-diagnostics

#include "Box.h"
#include "M1.h"
#include "M3.h"

void Run() {
  Box<> Present;
}
