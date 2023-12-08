// RUN: rm -rf %t
// RUN: %clang_cc1 -triple i686-unknown-unknown -fsyntax-only -fmodules -fimplicit-module-maps -fmodules-cache-path=%t %s -verify -ffreestanding
// RUN: %clang_cc1 -triple i686-unknown-unknown -fsyntax-only -fmodules -fmodule-map-file=%resource_dir/module.modulemap -fmodules-cache-path=%t %s -verify -ffreestanding
// RUN: %clang_cc1 -triple i686-unknown-unknown -fsyntax-only -fmodules -fimplicit-module-maps -fmodules-cache-path=%t %s -verify -ffreestanding -fno-gnu-inline-asm
// RUN: %clang_cc1 -triple i686--windows -fsyntax-only -fmodules -fimplicit-module-maps -fmodules-cache-path=%t %s -verify -ffreestanding -fno-gnu-inline-asm -fms-extensions -fms-compatibility-version=17.00
// expected-no-diagnostics

#include<x86intrin.h>

