/// Most likely platform specific sed differences
// UNSUPPORTED: system-windows

/// This test validates that modules that depend on prebuilt modules 
///   resolve `is-in-stable-directories` correctly. 
/// The steps are: 
/// 1. Scan dependencies to build the PCH. One of the module's depend on header 
///   that is seemingly from the sysroot. However, it depends on a local header that is overlaid.
/// 2. Build the PCH & dependency PCMs.
/// 3. Scan a source file that transitively depends on the same modules as the pcm.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/overlay.json.template > %t/overlay.json
// RUN: sed -e "s|DIR|%/t|g" %t/compile-pch.json.in > %t/compile-pch.json
// RUN: clang-scan-deps -compilation-database %t/compile-pch.json \
// RUN:   -j 1 -format experimental-full > %t/deps_pch.db
// RUN: %clang -x c-header -c %t/prebuild.h -isysroot %t/MacOSX.sdk \
// RUN:   -I%t/BuildDir -ivfsoverlay %t/overlay.json \
// RUN:   -I %t/MacOSX.sdk/usr/include -fmodules -fmodules-cache-path=%t/module-cache \
// RUN:   -fimplicit-module-maps -o %t/prebuild.pch
// RUN: sed -e "s|DIR|%/t|g" %t/compile-commands.json.in > %t/compile-commands.json
// RUN: clang-scan-deps -compilation-database %t/compile-commands.json \
// RUN:   -j 1 -format experimental-full > %t/deps.db
// RUN: cat %t/deps_pch.db | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t --check-prefix PCH_DEP
// RUN: cat %t/deps.db | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t  --check-prefix CLIENT

// PCH_DEP: "is-in-stable-directories": true
// PCH_DEP: "name": "A"

// PCH_DEP-NOT: "is-in-stable-directories": true

// Verify is-in-stable-directories is only assigned to the module that only depends on A.
// CLIENT-NOT: "is-in-stable-directories": true

// CLIENT: "name": "D"
// CLIENT: "is-in-stable-directories": true
// CLIENT: "name": "sys"

// CLIENT-NOT: "is-in-stable-directories": true

//--- compile-pch.json.in
[
{
    "directory": "DIR",
    "command": "clang -x c-header -c DIR/prebuild.h -isysroot DIR/MacOSX.sdk -IDIR/BuildDir -ivfsoverlay DIR/overlay.json -IDIR/MacOSX.sdk/usr/include -fmodules -fmodules-cache-path=DIR/module-cache -fimplicit-module-maps -o DIR/prebuild.pch",
    "file": "DIR/prebuild.h"
}
]

//--- compile-commands.json.in
[
{
    "directory": "DIR",
    "command": "clang -c DIR/client.c -isysroot DIR/MacOSX.sdk -IDIR/BuildDir -ivfsoverlay DIR/overlay.json -IDIR/MacOSX.sdk/usr/include -fmodules -fmodules-cache-path=DIR/module-cache -fimplicit-module-maps -include-pch DIR/prebuild.pch",
    "file": "DIR/client.c"
}
]

//--- overlay.json.template
{
  "version": 0,
  "case-sensitive": "false",
  "roots": [
    {
          "external-contents": "DIR/BuildDir/B_vfs.h",
          "name": "DIR/MacOSX.sdk/usr/include/B/B_vfs.h",
          "type": "file"
    }
  ]
}

//--- MacOSX.sdk/usr/include/A/module.modulemap
module A [system] {
  umbrella "."
}

//--- MacOSX.sdk/usr/include/A/A.h
typedef int A_type;

//--- MacOSX.sdk/usr/include/B/module.modulemap
module B [system] {
  umbrella "."
}

//--- MacOSX.sdk/usr/include/B/B.h
#include <B/B_vfs.h>

//--- BuildDir/B_vfs.h
typedef int local_t;

//--- MacOSX.sdk/usr/include/sys/sys.h
#include <A/A.h>
typedef int sys_t_m;

//--- MacOSX.sdk/usr/include/sys/module.modulemap
module sys [system] {
  umbrella "."
}

//--- MacOSX.sdk/usr/include/B_transitive/B.h
#include <B/B.h>

//--- MacOSX.sdk/usr/include/B_transitive/module.modulemap
module B_transitive [system] {
  umbrella "."
}

//--- MacOSX.sdk/usr/include/C/module.modulemap
module C [system] {
  umbrella "."
}

//--- MacOSX.sdk/usr/include/C/C.h
#include <B_transitive/B.h>


//--- MacOSX.sdk/usr/include/D/module.modulemap
module D [system] {
  umbrella "."
}

//--- MacOSX.sdk/usr/include/D/D.h
#include <C/C.h>

//--- prebuild.h
#include <A/A.h>
#include <C/C.h> // This dependency transitively depends on a local header.

//--- client.c
#include <sys/sys.h>
#include <D/D.h> // This dependency transitively depends on a local header.
