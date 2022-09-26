// There are some edge-cases where Clang depends on knowing the module whose implementation it's currently building.
// This test makes sure scanner always reports the corresponding module map.

// RUN: rm -rf %t
// RUN: split-file %s %t

//--- frameworks/A.framework/Modules/module.modulemap
framework module A { umbrella header "A.h" }
//--- frameworks/B.framework/Modules/module.modulemap
framework module B { umbrella header "B.h" }
//--- frameworks/A.framework/Headers/A.h
//--- frameworks/B.framework/Headers/B.h
//--- frameworks/A.framework/Modules/module.private.modulemap
framework module A_Private { umbrella header "A_Private.h" }
//--- frameworks/B.framework/Modules/module.private.modulemap
framework module B_Private { umbrella header "B_Private.h" }
//--- frameworks/A.framework/PrivateHeaders/A_Private.h
#import <A/H.h>
//--- frameworks/B.framework/PrivateHeaders/B_Private.h
#import <B/H.h>

//--- shared/H.h

//--- overlay.json.template
{
  "case-sensitive": "false",
  "version": 0,
  "roots": [
    {
      "contents": [
        {
          "external-contents": "DIR/shared/H.h",
          "name": "H.h",
          "type": "file"
        }
      ],
      "name": "DIR/frameworks/A.framework/PrivateHeaders",
      "type": "directory"
    },
    {
      "contents": [
        {
          "external-contents": "DIR/shared/H.h",
          "name": "H.h",
          "type": "file"
        }
      ],
      "name": "DIR/frameworks/B.framework/PrivateHeaders",
      "type": "directory"
    }
  ]
}

//--- cdb.json.template
[{
  "file": "DIR/tu.m",
  "directory": "DIR",
  "command": "clang -fmodules -fmodules-cache-path=DIR/cache -fmodule-name=A -ivfsoverlay DIR/overlay.json -F DIR/frameworks -c DIR/tu.m -o DIR/tu.o"
}]

//--- tu.m
@import B;
#import <A/H.h>
#import <B/H.h>

// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: sed -e "s|DIR|%/t|g" %t/overlay.json.template > %t/overlay.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full > %t/result.json
// RUN: cat %t/result.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t
// CHECK:      {
// CHECK:        "translation-units": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "commands": [
// CHECK-NEXT:         {
// CHECK:                "command-line": [
// CHECK:                  "-fmodule-map-file=[[PREFIX]]/frameworks/A.framework/Modules/module.modulemap",
// CHECK:                  "-fmodule-name=A",
// CHECK:                ],
// CHECK-NEXT:           "executable": "clang",
// CHECK-NEXT:           "file-deps": [
// CHECK-NEXT:             "[[PREFIX]]/tu.m",
// CHECK-NEXT:             "[[PREFIX]]/shared/H.h"
// CHECK-NEXT:           ],
// CHECK-NEXT:           "input-file": "[[PREFIX]]/tu.m"
// CHECK-NEXT:         }
// CHECK-NEXT:       ]
// CHECK-NEXT:     }
// CHECK-NEXT:   ]
// CHECK-NEXT: }

// RUN: %deps-to-rsp %t/result.json --module-name=B > %t/B.cc1.rsp
// RUN: %clang @%t/B.cc1.rsp

// RUN: %deps-to-rsp %t/result.json --tu-index=0 > %t/tu.rsp
// RUN: %clang @%t/tu.rsp
