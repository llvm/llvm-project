// Check that a relocated module is rebuilt and no error occurs. 
// This is required when a incremental build adds a newer version of an 
//    already built library into a preexisting search path.
// In this example, on the second scan, `DepThatLoadsOldPCMs` is resolved first and 
//    populates `MovedDep` into memory. 
//  Then when `InvalidatedDep` is loaded, it's input file is out of date 
//    and requires a rebuild.
// When that happens the compiler notices `MovedDep` is in a earlier search path.


// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/compile-commands.json.in > %t/compile-commands.json

// RUN: touch %t/session.timestamp
// RUN: clang-scan-deps -format experimental-full -j 1 \
// RUN:   -compilation-database %t/compile-commands.json -o %t/deps1.json 
// RUN: cat %t/deps1.json | FileCheck %s --check-prefix=DEPS1 

// Model update where same framework appears in earlier search path.
// This can occur on an incremental build where dependency relationships are updated.
// RUN: touch %t/session.timestamp
// RUN: sleep 1
// RUN: mkdir %t/preferred_frameworks/
// RUN: cp -r %t/fallback_frameworks/MovedDep.framework %t/preferred_frameworks/
// RUN: touch %t/fallback_frameworks/InvalidatedDep.framework/Modules/module.modulemap

// RUN: clang-scan-deps -format experimental-full -j 1 \
// RUN:   -compilation-database %t/compile-commands.json -o %t/deps2.json
// RUN: cat %t/deps2.json | FileCheck %s --check-prefix=DEPS2

// DEPS1: "clang-module-deps": [],
// DEPS1-NEXT: "clang-modulemap-file": "{{.*}}fallback_frameworks{{.*}}MovedDep.framework
// DEPS1: "name": "MovedDep"

// DEPS2: "clang-module-deps": [],
// DEPS2-NEXT: "clang-modulemap-file": "{{.*}}preferred_frameworks{{.*}}MovedDep.framework
// DEPS2: "name": "MovedDep"

//--- compile-commands.json.in
[
{
  "directory": "DIR",
  "command": "clang -c DIR/tu1.c -fmodules -fmodules-cache-path=DIR/cache -FDIR/preferred_frameworks -FDIR/fallback_frameworks  -fbuild-session-file=DIR/session.timestamp -fmodules-validate-once-per-build-session -o DIR/tu1.o ",
  "file": "DIR/tu1.c"                                                                       
}
]

//--- fallback_frameworks/MovedDep.framework/Modules/module.modulemap
framework module MovedDep { header "MovedDep.h" }
//--- fallback_frameworks/MovedDep.framework/Headers/MovedDep.h
int foo(void);

//--- fallback_frameworks/InvalidatedDep.framework/Modules/module.modulemap
framework module InvalidatedDep { header "InvalidatedDep.h" }
//--- fallback_frameworks/InvalidatedDep.framework/Headers/InvalidatedDep.h
#include <MovedDep/MovedDep.h>

//--- fallback_frameworks/DirectDep.framework/Modules/module.modulemap
framework module DirectDep { header "DirectDep.h" }
//--- fallback_frameworks/DirectDep.framework/Headers/DirectDep.h
#include <DepThatLoadsOldPCMs/DepThatLoadsOldPCMs.h>
#include <InvalidatedDep/InvalidatedDep.h>

//--- fallback_frameworks/DepThatLoadsOldPCMs.framework/Modules/module.modulemap
framework module DepThatLoadsOldPCMs { header "DepThatLoadsOldPCMs.h" }
//--- fallback_frameworks/DepThatLoadsOldPCMs.framework/Headers/DepThatLoadsOldPCMs.h
#include <MovedDep/MovedDep.h>

//--- tu1.c
#include <DirectDep/DirectDep.h>
