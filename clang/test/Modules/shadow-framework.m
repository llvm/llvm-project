// RUN: rm -rf %t
// RUN: split-file %s %t

// This test checks that redefinitions of frameworks are ignored.

//--- include/module.modulemap
module first { header "first.h" }
module FW {}
//--- include/first.h

//--- frameworks/FW.framework/Modules/module.modulemap
framework module FW { header "FW.h" }
//--- frameworks/FW.framework/Headers/FW.h

//--- tu.c
#import "first.h" // expected-remark {{importing module 'first'}}
#import <FW/FW.h>

// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t/cache -fimplicit-module-maps \
// RUN:   -I %t/include -F %t/frameworks -fsyntax-only %t/tu.c -Rmodule-import -verify
