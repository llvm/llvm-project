// This test verifies that the modules visible to the translation unit are computed in dependency scanning.
// "client" in the first scan represents the translation unit that imports an explicit submodule, 
//    that only exports one other module. 
// In the second scan, the translation unit that imports an explicit submodule, 
//    that exports an additional module. 
// Thus, the dependencies of the top level module for the submodule always differ from what is visible to the TU.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/compile-commands.json.in > %t/compile-commands.json
// RUN: clang-scan-deps -emit-visible-modules -compilation-database %t/compile-commands.json \
// RUN:   -j 1 -format experimental-full 2>&1 > %t/result-first-scan.json
// RUN: cat %t/result-first-scan.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t --check-prefix=SINGLE

/// Re-run scan with different module map for direct dependency.
// RUN: mv %t/A_with_visible_export.modulemap %t/Sysroot/usr/include/A/module.modulemap
// RUN: clang-scan-deps -emit-visible-modules -compilation-database %t/compile-commands.json \
// RUN:   -j 1 -format experimental-full 2>&1 > %t/result.json
// RUN: cat %t/result.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t --check-prefix=MULTIPLE

// RUN: %deps-to-rsp %t/result.json --module-name=transitive > %t/transitive.rsp
// RUN: %deps-to-rsp %t/result.json --module-name=visible > %t/visible.rsp
// RUN: %deps-to-rsp %t/result.json --module-name=invisible > %t/invisible.rsp
// RUN: %deps-to-rsp %t/result.json --module-name=A > %t/A.rsp
// RUN: %deps-to-rsp %t/result.json --tu-index=0 > %t/tu.rsp

// RUN: %clang @%t/transitive.rsp
// RUN: %clang @%t/visible.rsp
// RUN: %clang @%t/invisible.rsp
// RUN: %clang @%t/A.rsp

/// Verify compilation & scan agree with each other.
// RUN: not %clang @%t/tu.rsp -o %t/blah.o 2>&1 | FileCheck %s --check-prefix=COMPILE

// SINGLE:        "visible-clang-modules": [
// SINGLE-NEXT:     "A"
// SINGLE-NEXT:   ]

// MULTIPLE:        "visible-clang-modules": [
// MULTIPLE-NEXT:     "A",
// MULTIPLE-NEXT:     "visible"
// MULTIPLE-NEXT:   ]

// COMPILE-NOT:   'visible_t' must be declared before it is used
// COMPILE:       'transitive_t' must be declared before it is used
// COMPILE:       'invisible_t' must be declared before it is used

//--- compile-commands.json.in
[
{
  "directory": "DIR",
  "command": "clang -c DIR/client.c -isysroot DIR/Sysroot -IDIR/Sysroot/usr/include -fmodules -fmodules-cache-path=DIR/module-cache -fimplicit-module-maps",
  "file": "DIR/client.c"
}
]

//--- Sysroot/usr/include/A/module.modulemap
module A {
  explicit module visibleToTU {
    header "visibleToTU.h"
  }
  explicit module invisibleToTU {
    header "invisibleToTU.h" 
  }
}

//--- A_with_visible_export.modulemap
module A {
  explicit module visibleToTU {
    header "visibleToTU.h"
    export visible
  }
  explicit module invisibleToTU {
    header "invisibleToTU.h" 
  }
}

//--- Sysroot/usr/include/A/visibleToTU.h
#include <visible/visible.h>
typedef int A_visibleToTU;

//--- Sysroot/usr/include/A/invisibleToTU.h
#include <invisible/invisible.h>
typedef int A_invisibleToTU;

//--- Sysroot/usr/include/invisible/module.modulemap
module invisible {
  umbrella "."
}

//--- Sysroot/usr/include/invisible/invisible.h
typedef int invisible_t;

//--- Sysroot/usr/include/visible/module.modulemap
module visible {
  umbrella "."
}

//--- Sysroot/usr/include/visible/visible.h
#include <transitive/transitive.h>
typedef int visible_t;

//--- Sysroot/usr/include/transitive/module.modulemap
module transitive {
  umbrella "."
}

//--- Sysroot/usr/include/transitive/transitive.h
typedef int transitive_t;

//--- client.c
#include <A/visibleToTU.h> 
visible_t foo_v(void);
// Both decls are not visible, thus should fail to actually compile.
transitive_t foo_t(void);
invisible_t foo_i(void); 
