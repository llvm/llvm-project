// UNSUPPORTED: system-windows
// RUN: rm -rf %t
// RUN: split-file %s %t

// Verify the stable dir path.
//--- Sysroot/usr/include/SysA/module.modulemap
module SysA {
  header "SysA.h"
}

//--- Sysroot/usr/include/SysA/SysA.h
int SysVal = 42;

//--- cdb.json.template
[{
  "file": "",
  "directory": "DIR",
  "command": "clang -fmodules -fmodules-cache-path=DIR/cache -isysroot DIR/Sysroot -IDIR/Sysroot/usr/include -x c"
}]

// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full -module-names=SysA > %t/result.json
// RUN: cat %t/result.json | sed 's:\\\\\?:/:g' | FileCheck -DPREFIX=%/t %s

// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "is-in-stable-directories": true,
// CHECK-NEXT:       "clang-module-deps": [],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/Sysroot/usr/include/SysA/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "{{.*}}",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/Sysroot/usr/include/SysA/module.modulemap",
// CHECK-NEXT:         "[[PREFIX]]/Sysroot/usr/include/SysA/SysA.h"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "link-libraries": [],
// CHECK-NEXT:       "name": "SysA"
// CHECK-NEXT:     }
// CHECK-NEXT:   ],
// CHECK-NEXT:   "translation-units": []
// CHECK-NEXT: }
