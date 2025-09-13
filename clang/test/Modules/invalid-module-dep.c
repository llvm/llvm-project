/// This tests the expected error case when there is a mismatch between the pcm dependencies passed in 
/// the command line with `fmodule-file` and whats encoded in the pcm. 

/// The steps are: 
/// 1. Build the module A with no dependencies. The first variant to build is A-1.pcm.
/// 2. Build the same module with files that resolve from different search paths. 
///     This variant is named A-2.pcm.
/// 3. Build module B that depends on the earlier module A-1.pcm.
/// 4. Build client that directly depends on both modules (A & B), 
///     but depends on a incompatible variant of A (A-2.pcm) for B to use.

// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -emit-module -x c -fmodules -fno-implicit-modules -isysroot %t/Sysroot \
// RUN:     -I%t/Sysroot/usr/include \
// RUN:     -fmodule-name=A %t/Sysroot/usr/include/A/module.modulemap -o %t/A-1.pcm

// RUN: %clang_cc1 -emit-module -x c -fmodules -fno-implicit-modules -isysroot %t/Sysroot \
// RUN:     -I%t/BuildDir \
// RUN:     -fmodule-name=A %t/BuildDir/A/module.modulemap -o %t/A-2.pcm

// RUN: %clang_cc1 -emit-module -x c -fmodules -fno-implicit-modules -isysroot %t/Sysroot \
// RUN:     -I%t/Sysroot/usr/include \
// RUN:     -fmodule-map-file=%t/Sysroot/usr/include/A/module.modulemap \
// RUN:     -fmodule-file=A=%t/A-1.pcm \
// RUN:     -fmodule-name=B %t/Sysroot/usr/include/B/module.modulemap -o %t/B-1.pcm

// RUN: %clang_cc1 -x c -fmodules -fno-implicit-modules -isysroot %t/Sysroot \
// RUN:     -I%t/BuildDir -I%t/Sysroot/usr/include \
// RUN:     -fmodule-map-file=%t/BuildDir/A/module.modulemap \
// RUN:     -fmodule-map-file=%t/Sysroot/usr/include/B/module.modulemap \
// RUN:     -fmodule-file=A=%t/A-2.pcm -fmodule-file=B=%t/B-1.pcm \
// RUN:     -Wmodule-file-mapping-mismatch -verify %s  


#include <A/A.h>
#include <B/B.h> // expected-warning {{conflicts with imported file}} \
                 // expected-note {{imported by module 'B'}} \
                 // expected-error {{out of date and needs to be rebuilt}}

//--- Sysroot/usr/include/A/module.modulemap
module A [system] {
  umbrella header "A.h"
}
//--- Sysroot/usr/include/A/A.h
typedef int A_t;

//--- Sysroot/usr/include/B/module.modulemap
module B [system] {
  umbrella header "B.h"
}

//--- Sysroot/usr/include/B/B.h
#include <A/A.h>
typedef int B_t;

//--- BuildDir/A/module.modulemap
module A [system] {
  umbrella header "A.h"
}

//--- BuildDir/A/A.h
typedef int A_t;
