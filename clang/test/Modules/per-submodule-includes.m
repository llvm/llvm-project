// RUN: rm -rf %t
// RUN: split-file %s %t

//--- frameworks/Textual.framework/Headers/Header.h
static int symbol;

//--- frameworks/FW.framework/Modules/module.modulemap
framework module FW {
  umbrella header "FW.h"
  export *
  module * { export * }
}
//--- frameworks/FW.framework/Headers/FW.h
#import <FW/Sub1.h>
#import <FW/Sub2.h>
//--- frameworks/FW.framework/Headers/Sub1.h
//--- frameworks/FW.framework/Headers/Sub2.h
#import <Textual/Header.h>

//--- pch.modulemap
module __PCH {
  header "pch.h"
  export *
}
//--- pch.h
#import <FW/Sub1.h>

//--- tu.m
#import <Textual/Header.h>
int fn() { return symbol; }

// Compilation using the PCH regularly succeeds. The import of FW/Sub1.h in the
// PCH is treated textually due to -fmodule-name=FW.
//
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t/cache -fimplicit-module-maps -F %t/frameworks -fmodule-name=FW \
// RUN:   -emit-pch -x objective-c %t/pch.h -o %t/pch.h.gch
//
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t/cache -fimplicit-module-maps -F %t/frameworks -fmodule-name=FW \
// RUN:   -include-pch %t/pch.h.gch -fsyntax-only %t/tu.m

// Compilation using the PCH as precompiled module fails. The import of FW/Sub1.h
// in the PCH is translated to an import. Nothing is preventing that now that
// -fmodule-name=FW has been replaced with -fmodule-name=__PCH.
//
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t/cache -fimplicit-module-maps -F %t/frameworks \
// RUN:   -emit-module -fmodule-name=__PCH -x objective-c %t/pch.modulemap -o %t/pch.h.pcm
//
// Loading FW.pcm marks Textual/Header.h as imported (because it is imported in
// FW.Sub2), so the TU does not import it again. It's contents remain invisible,
// though.
//
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t/cache -fimplicit-module-maps -F %t/frameworks \
// RUN:   -include %t/pch.h -fmodule-map-file=%t/pch.modulemap -fsyntax-only %t/tu.m
