// RUN: rm -rf %t
// RUN: %clang_cc1 -fsyntax-only -fmodules -fimplicit-module-maps -fmodules-cache-path=%t %s -verify -ffreestanding
// expected-no-diagnostics

#include<x86intrin.h>

