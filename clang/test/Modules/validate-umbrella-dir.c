// Adding a header to an umbrella directory (or to an umbrella header's
// directory) is a "negative dependency": a clean build would pick it up, but
// ordinary input-file validation only re-stats headers that already were
// inputs. Check that -fmodules-validate-umbrella-dirs catches it.

// RUN: rm -rf %t
// RUN: split-file %s %t

//===----------------------------------------------------------------------===//
// Umbrella directory.
//===----------------------------------------------------------------------===//

//--- dir/module.modulemap
module Umbrella {
  umbrella "umbrella"
  exclude header "umbrella/excluded.h"
}
//--- dir/umbrella/a.h
//--- dir/umbrella/excluded.h
#error "this header is excluded and should never be compiled"
//--- dir/tu.c
#include "umbrella/a.h"

// RUN: %clang_cc1 -fsyntax-only -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/dir/cache -I %t/dir \
// RUN:   -Rmodule-build %t/dir/tu.c 2>&1 | FileCheck %s --check-prefix=BUILD -DNAME=Umbrella

// No changes: up to date with or without the flag.
//
// RUN: %clang_cc1 -fsyntax-only -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/dir/cache -I %t/dir \
// RUN:   -Rmodule-build %t/dir/tu.c 2>&1 | FileCheck %s --check-prefix=NOBUILD --allow-empty -DNAME=Umbrella
// RUN: %clang_cc1 -fsyntax-only -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/dir/cache -I %t/dir -fmodules-validate-umbrella-dirs \
// RUN:   -Rmodule-build %t/dir/tu.c 2>&1 | FileCheck %s --check-prefix=NOBUILD --allow-empty -DNAME=Umbrella

// Touching an excluded header must not rebuild: a clean build ignores it.
// (The sleep makes later additions strictly newer than the module file.)
//
// RUN: sleep 1
// RUN: touch %t/dir/umbrella/excluded.h
// RUN: %clang_cc1 -fsyntax-only -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/dir/cache -I %t/dir -fmodules-validate-umbrella-dirs \
// RUN:   -Rmodule-build %t/dir/tu.c 2>&1 | FileCheck %s --check-prefix=NOBUILD --allow-empty -DNAME=Umbrella

// RUN: touch %t/dir/umbrella/b.h

// Adding b.h rebuilds only under the flag.
//
// RUN: %clang_cc1 -fsyntax-only -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/dir/cache -I %t/dir \
// RUN:   -Rmodule-build %t/dir/tu.c 2>&1 | FileCheck %s --check-prefix=NOBUILD --allow-empty -DNAME=Umbrella
// RUN: %clang_cc1 -fsyntax-only -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/dir/cache -I %t/dir -fmodules-validate-umbrella-dirs \
// RUN:   -Rmodule-build -Rmodule-validation %t/dir/tu.c 2>&1 | FileCheck %s --check-prefixes=BUILD,DIR-REMARK -DNAME=Umbrella

// Converges: the rebuilt module is newer than b.h, so no further rebuild.
//
// RUN: %clang_cc1 -fsyntax-only -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/dir/cache -I %t/dir -fmodules-validate-umbrella-dirs \
// RUN:   -Rmodule-build %t/dir/tu.c 2>&1 | FileCheck %s --check-prefix=NOBUILD --allow-empty -DNAME=Umbrella

//===----------------------------------------------------------------------===//
// Umbrella header. A new sibling changes -Wincomplete-umbrella, so a clean
// build differs and the module should rebuild under the flag.
//===----------------------------------------------------------------------===//

//--- hdr/module.modulemap
module UH { umbrella header "UH.h" }
//--- hdr/UH.h
#include "a.h"
//--- hdr/a.h
//--- hdr/tu.c
#include "UH.h"

// RUN: %clang_cc1 -fsyntax-only -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/hdr/cache -I %t/hdr \
// RUN:   -Rmodule-build %t/hdr/tu.c 2>&1 | FileCheck %s --check-prefix=BUILD -DNAME=UH

// RUN: sleep 1
// RUN: touch %t/hdr/b.h

// Adding a sibling rebuilds only under the flag.
//
// RUN: %clang_cc1 -fsyntax-only -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/hdr/cache -I %t/hdr \
// RUN:   -Rmodule-build %t/hdr/tu.c 2>&1 | FileCheck %s --check-prefix=NOBUILD --allow-empty -DNAME=UH
// RUN: %clang_cc1 -fsyntax-only -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/hdr/cache -I %t/hdr -fmodules-validate-umbrella-dirs \
// RUN:   -Rmodule-build -Rmodule-validation %t/hdr/tu.c 2>&1 | FileCheck %s --check-prefixes=BUILD,HDR-REMARK -DNAME=UH

// Converges.
//
// RUN: %clang_cc1 -fsyntax-only -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/hdr/cache -I %t/hdr -fmodules-validate-umbrella-dirs \
// RUN:   -Rmodule-build %t/hdr/tu.c 2>&1 | FileCheck %s --check-prefix=NOBUILD --allow-empty -DNAME=UH

//===----------------------------------------------------------------------===//
// Transitive module: the TU imports Top, which pulls in Leaf. Adding to Leaf's
// umbrella must invalidate Leaf even though the TU never names it (its module
// map is only parsed via the umbrella check's own lookup).
//===----------------------------------------------------------------------===//

//--- trans/module.modulemap
module Top { umbrella "top" export * }
module Leaf { umbrella "leaf" export * }
//--- trans/top/top.h
#include "leaf/leaf.h"
//--- trans/leaf/leaf.h
//--- trans/tu.c
#include "top/top.h"

// RUN: %clang_cc1 -fsyntax-only -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/trans/cache -I %t/trans \
// RUN:   -Rmodule-build %t/trans/tu.c 2>&1 | FileCheck %s --check-prefix=BUILD -DNAME=Leaf

// RUN: sleep 1
// RUN: touch %t/trans/leaf/leaf2.h
// RUN: %clang_cc1 -fsyntax-only -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/trans/cache -I %t/trans -fmodules-validate-umbrella-dirs \
// RUN:   -Rmodule-build -Rmodule-validation %t/trans/tu.c 2>&1 | FileCheck %s --check-prefixes=BUILD,TRANS-REMARK -DNAME=Leaf

// DIR-REMARK: module 'Umbrella' is out of date because header '{{.*}}b.h' was added to its umbrella directory
// HDR-REMARK: module 'UH' is out of date because header '{{.*}}b.h' was added to its umbrella header's directory
// TRANS-REMARK: module 'Leaf' is out of date because header '{{.*}}leaf2.h' was added to its umbrella directory

// BUILD: building module '[[NAME]]'
// NOBUILD-NOT: building module '[[NAME]]'
