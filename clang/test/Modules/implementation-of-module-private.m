// RUN: rm -rf %t
// RUN: split-file %s %t

//--- frameworks/FW.framework/Modules/module.modulemap
framework module FW {}
//--- frameworks/FW.framework/Modules/module.private.modulemap
framework module FW_Private {}

//--- tu.m
@import FW_Private; // expected-error{{@import of module 'FW_Private' in implementation of 'FW'; use #import}}

// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t/cache -fimplicit-module-maps \
// RUN:   -fmodule-name=FW -F %t/frameworks %t/tu.m -verify
