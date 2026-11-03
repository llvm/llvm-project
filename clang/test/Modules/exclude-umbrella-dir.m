// RUN: rm -rf %t
// RUN: split-file %s %t

// A directory excluded via `exclude umbrella` is left out of the umbrella
// module, while other directories under the same umbrella still resolve. If the
// excluded directory were built into the module, secret.h's #error would fire.
// RUN: %clang_cc1 -x objective-c -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/cache -I%t/basic %t/basic/tu.m -verify

// An exclude in one module does not remove a directory from a different module
// whose umbrella also covers it.
// RUN: %clang_cc1 -x objective-c -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/cache-scope -I%t/scope %t/scope/tu.m -verify

// A nonexistent excluded directory is a warning, not an error: the module
// still builds.
// RUN: %clang_cc1 -fmodules -fmodule-name=Missing -x c++-module-map \
// RUN:   %t/missing/module.modulemap -emit-module -o /dev/null -verify

// The directive round-trips through the module map printer.
// RUN: %clang_cc1 -fmodules -fmodule-name=RoundTrip -x c++-module-map \
// RUN:   %t/roundtrip/module.modulemap -E | FileCheck %t/roundtrip/module.modulemap

// A header under an excluded directory is treated like an `exclude header`
// header: including it from the module's own headers is not reported as a
// non-modular include.
// RUN: %clang_cc1 -x objective-c -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/cache-quiet -I%t/quiet \
// RUN:   -Wnon-modular-include-in-module -Werror %t/quiet/tu.m -verify

//--- basic/module.modulemap
module Basic {
  umbrella "inc"
  exclude umbrella "inc/private"
}

//--- basic/inc/root.h
typedef int basic_root;

//--- basic/inc/pub/pub.h
typedef int basic_pub;

//--- basic/inc/private/secret.h
#error secret.h must be excluded from module Basic

//--- basic/tu.m
@import Basic;
basic_root use_root;
basic_pub use_pub;
// expected-no-diagnostics

//--- scope/module.modulemap
module Excluder {
  umbrella "excluder"
  exclude umbrella "shared"
}
module Sharer {
  umbrella "shared"
}

//--- scope/excluder/e.h
typedef int excluder_e;

//--- scope/shared/s.h
typedef int sharer_s;

//--- scope/tu.m
// Sharer's umbrella covers "shared" even though Excluder excludes it, so
// sharer_s is part of Sharer and visible after the import.
@import Sharer;
sharer_s use_s;
// expected-no-diagnostics

//--- missing/module.modulemap
module Missing {
  umbrella "inc"
  exclude umbrella "inc/does-not-exist"
}
#pragma clang module contents
// expected-warning@3 {{excluded directory 'inc/does-not-exist' not found}}

//--- missing/inc/h.h
typedef int missing_h;

//--- roundtrip/module.modulemap
// CHECK: module RoundTrip {
// CHECK:   umbrella "inc"
// CHECK:   exclude umbrella "inc/priv"
module RoundTrip {
  umbrella "inc"
  exclude umbrella "inc/priv"
}

//--- roundtrip/inc/a.h
typedef int rt_a;

//--- roundtrip/inc/priv/b.h
typedef int rt_b;

//--- quiet/module.modulemap
module Quiet {
  umbrella "inc"
  exclude umbrella "inc/private"
}

//--- quiet/inc/pub.h
#include "inc/private/priv.h"
typedef int quiet_pub;

//--- quiet/inc/private/priv.h
typedef int quiet_priv;

//--- quiet/tu.m
@import Quiet;
quiet_pub use_pub;
// expected-no-diagnostics
