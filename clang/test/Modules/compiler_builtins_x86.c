// RUN: rm -rf %t
// RUN: %clang_cc1 -triple i686-unknown-unknown -fsyntax-only -fmodules -fimplicit-module-maps -fmodules-cache-path=%t %s -verify -ffreestanding
// RUN: %clang_cc1 -triple i686-unknown-unknown -fsyntax-only -fmodules -fmodule-map-file=%resource_dir/module.modulemap -fmodules-cache-path=%t %s -verify -ffreestanding
// expected-no-diagnostics

#include<x86intrin.h>

