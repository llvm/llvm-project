// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs/self-referencing-lambda %s -verify -emit-obj -std=c++14 -o %t2.o
// expected-no-diagnostics

#include "a.h"
