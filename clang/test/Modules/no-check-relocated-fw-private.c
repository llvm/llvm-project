// RUN: rm -rf %t
// RUN: split-file %s %t

//--- frameworks1/FW1.framework/Modules/module.modulemap
framework module FW1 { header "FW1.h" }
//--- frameworks1/FW1.framework/Headers/FW1.h
#import <FW2/FW2.h>

//--- frameworks2/FW2.framework/Modules/module.modulemap
framework module FW2 { header "FW2.h" }
//--- frameworks2/FW2.framework/Modules/module.private.modulemap
framework module FW2_Private { header "FW2_Private.h" }
//--- frameworks2/FW2.framework/Headers/FW2.h
//--- frameworks2/FW2.framework/PrivateHeaders/FW2_Private.h

//--- tu.c
#import <FW1/FW1.h>         // expected-remark{{importing module 'FW1'}} \
                            // expected-remark{{importing module 'FW2'}}
#import <FW2/FW2_Private.h> // expected-remark{{importing module 'FW2_Private'}}

// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t/cache -fimplicit-module-maps \
// RUN:   -F %t/frameworks1 -F %t/frameworks2 -fsyntax-only %t/tu.c \
// RUN:   -fno-modules-check-relocated -Rmodule-import -verify
