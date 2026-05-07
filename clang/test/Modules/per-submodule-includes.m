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

// The import of FW/Sub1.h in the PCH is treated textually due to -fmodule-name=FW.
// Because of this, pch.h.pch only imports an empty file (FW/Sub1.h), not
// the module FW. Therefore importing Textual/Header.h just works.
//
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t/cache -fimplicit-module-maps -F %t/frameworks -fmodule-name=FW \
// RUN:   -emit-pch -x objective-c %t/pch.h -o %t/pch.h.pch
//
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t/cache -fimplicit-module-maps -F %t/frameworks -fmodule-name=FW \
// RUN:   -include-pch %t/pch.h.pch -fsyntax-only %t/tu.m

// Building the __PCH module treats #import <FW/Sub1.h> as a module import from
// pch.h. Such a module import triggers a build of the module FW. Since FW/Sub2.h
// imports Textual/Header.h, Textual/Header.h becomes visible in submodule Sub2.
//
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t/cache -fimplicit-module-maps -F %t/frameworks \
// RUN:   -emit-module -fmodule-name=__PCH -x objective-c %t/pch.modulemap -o %t/pch.h.pcm
//
// Loading FW.pcm marks Textual/Header.h as imported (because it is imported in
// FW.Sub2), so the TU does not import it again.
//
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t/cache -fimplicit-module-maps -F %t/frameworks \
// RUN:   -include %t/pch.h -fmodule-map-file=%t/pch.modulemap -fsyntax-only %t/tu.m

// With local submodule visibility, importing FW.Sub1 does NOT make FW.Sub2
// visible. Textual/Header.h was included by Sub2, which is not visible, so
// #import re-enters the header textually making its contents available.
//
// RUN: rm -rf %t/cache
// RUN: %clang_cc1 -fmodules -fmodules-local-submodule-visibility -fmodules-cache-path=%t/cache -fimplicit-module-maps -F %t/frameworks \
// RUN:   -emit-module -fmodule-name=__PCH -x objective-c %t/pch.modulemap -o %t/pch.lsv.pcm
//
// RUN: %clang_cc1 -fmodules -fmodules-local-submodule-visibility -fmodules-cache-path=%t/cache -fimplicit-module-maps -F %t/frameworks \
// RUN:   -include %t/pch.h -fmodule-map-file=%t/pch.modulemap -fsyntax-only %t/tu.m
