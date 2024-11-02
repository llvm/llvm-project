// RUN: rm -rf %t && mkdir %t
// RUN: split-file %s %t

//--- a/module.modulemap
module a {}

//--- b/module.modulemap
module b {}

//--- c/module.modulemap
module c {}

//--- module.modulemap
module m { header "m.h" }
//--- m.h
@import c;

//--- test-simple.m
// expected-no-diagnostics
@import m;

// Build modules with the non-affecting "a/module.modulemap".
// RUN: %clang_cc1 -I %t/a -I %t/b -I %t/c -I %t -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache -fdisable-module-hash %t/test-simple.m -verify
// RUN: mv %t/cache %t/cache-with

// Build modules without the non-affecting "a/module.modulemap".
// RUN: rm -rf %t/a/module.modulemap
// RUN: %clang_cc1 -I %t/a -I %t/b -I %t/c -I %t -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache -fdisable-module-hash %t/test-simple.m -verify
// RUN: mv %t/cache %t/cache-without

// Check that the PCM files are bit-for-bit identical.
// RUN: diff %t/cache-with/m.pcm %t/cache-without/m.pcm
