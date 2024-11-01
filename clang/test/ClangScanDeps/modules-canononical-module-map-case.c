// This test checks that VFS-mapped module map path has the correct spelling
// after canonicalization, even if it was first accessed using different case.

// RUN: rm -rf %t
// RUN: split-file %s %t

//--- actual/One.h
#import <FW/Two.h>
//--- actual/Two.h
// empty
//--- frameworks/FW.framework/Modules/module.modulemap
framework module FW {
  header "One.h"
  header "Two.h"
}
//--- tu.m
#import <fw/One.h>

//--- overlay.json.in
{
  "version": 0,
  "case-sensitive": "false",
  "roots": [
    {
      "contents": [
        {
          "external-contents": "DIR/actual/One.h",
          "name": "One.h",
          "type": "file"
        },
        {
          "external-contents": "DIR/actual/Two.h",
          "name": "Two.h",
          "type": "file"
        }
      ],
      "name": "DIR/frameworks/FW.framework/Headers",
      "type": "directory"
    }
  ]
}

//--- cdb.json.in
[{
  "directory": "DIR",
  "file": "DIR/tu.m",
  "command": "clang -fmodules -fmodules-cache-path=DIR/cache -ivfsoverlay DIR/overlay.json -F DIR/frameworks -c DIR/tu.m -o DIR/tu.o"
}]

// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.in > %t/cdb.json
// RUN: sed -e "s|DIR|%/t|g" %t/overlay.json.in > %t/overlay.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full > %t/result.json
// RUN: cat %t/result.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t

// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/frameworks/FW.framework/Modules/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK:              "-x"
// CHECK-NEXT:         "objective-c"
// CHECK-NEXT:         "[[PREFIX]]/frameworks/FW.framework/Modules/module.modulemap"
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "{{.*}}",
// CHECK-NEXT:       "file-deps": [
// CHECK:            ],
// CHECK-NEXT:       "name": "FW"
// CHECK-NEXT:     }
// CHECK-NEXT:   ]
// CHECK:      }
