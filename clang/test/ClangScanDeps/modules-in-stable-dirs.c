// Most likely platform specific sed differences
// UNSUPPORTED: system-windows

// This test verifies modules that are entirely comprised from stable directory inputs are captured in
// dependency information.

// The first compilation verifies that transitive dependencies on local input are captured.
// The second compilation verifies that external paths are resolved when a 
// vfsoverlay for determining is-in-stable-directories.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/compile-commands.json.in > %t/compile-commands.json
// RUN: sed -e "s|DIR|%/t|g" %t/overlay.json.template > %t/overlay.json
// RUN: clang-scan-deps -compilation-database %t/compile-commands.json \
// RUN:   -j 1 -format experimental-full > %t/deps.db
// RUN: cat %t/deps.db | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t

// CHECK:   "modules": [
// CHECK-NEXT:     {
// CHECK:            "is-in-stable-directories": true,
// CHECK:            "name": "A"

// Verify that there are no more occurances of sysroot.
// CHECK-NOT:            "is-in-stable-directories"

// CHECK:            "name": "A"
// CHECK:            "USE_VFS"
// CHECK:            "name": "B"
// CHECK:            "name": "C"
// CHECK:            "name": "D"
// CHECK:            "name": "NotInSDK"

//--- compile-commands.json.in
[
{
  "directory": "DIR",
  "command": "clang -c DIR/client.c -isysroot DIR/Sysroot -IDIR/Sysroot/usr/include -IDIR/BuildDir -fmodules -fmodules-cache-path=DIR/module-cache -fimplicit-module-maps",
  "file": "DIR/client.c"
},
{
  "directory": "DIR",
  "command": "clang -c DIR/client.c -isysroot DIR/Sysroot -IDIR/Sysroot/usr/include -ivfsoverlay DIR/overlay.json -DUSE_VFS -IDIR/BuildDir -fmodules -fmodules-cache-path=DIR/module-cache -fimplicit-module-maps",
  "file": "DIR/client.c"
}
]

//--- overlay.json.template
{
  "version": 0,
  "case-sensitive": "false",
  "roots": [
    {
          "external-contents": "DIR/SysrootButNotReally/A/A_vfs.h",
          "name": "DIR/Sysroot/usr/include/A/A_vfs.h",
          "type": "file"
    }
  ]
}

//--- Sysroot/usr/include/A/module.modulemap
module A {
  umbrella "."
}

//--- Sysroot/usr/include/A/A.h
#ifdef USE_VFS
#include <A/A_vfs.h>
#endif 
typedef int A_t;

//--- SysrootButNotReally/A/A_vfs.h
typedef int typeFromVFS;

//--- Sysroot/usr/include/B/module.modulemap
module B [system] {
  umbrella "."
}

//--- Sysroot/usr/include/B/B.h
#include <C/C.h>
typedef int B_t;

//--- Sysroot/usr/include/C/module.modulemap
module C [system] {
  umbrella "."
}

//--- Sysroot/usr/include/C/C.h
#include <D/D.h>

//--- Sysroot/usr/include/D/module.modulemap
module D [system] {
  umbrella "."
}

// Simulate a header that will be resolved in a local directory, from a sysroot header.
//--- Sysroot/usr/include/D/D.h
#include <HeaderNotFoundInSDK.h>

//--- BuildDir/module.modulemap
module NotInSDK [system] {
  umbrella "."
}

//--- BuildDir/HeaderNotFoundInSDK.h
typedef int local_t;

//--- client.c
#include <A/A.h>
#include <B/B.h>
