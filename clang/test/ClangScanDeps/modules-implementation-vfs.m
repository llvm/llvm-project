// Ensure we get the virtual module map path for a module whose implementation
// file we are compiling.

// RUN: rm -rf %t
// RUN: split-file %s %t

//--- real/A.modulemap
framework module A { umbrella header "A.h" }
//--- real/A.private.modulemap
framework module A_Private { umbrella header "A_Private.h" }

//--- frameworks/A.framework/Headers/A.h
struct A { int x; };
//--- frameworks/A.framework/PrivateHeaders/A_Private.h
#import <A/A.h>

//--- frameworks/B.framework/Headers/B.h
#import <A/A.h>
//--- frameworks/B.framework/Modules/module.modulemap
framework module B { umbrella header "B.h" }

//--- overlay.json.template
{
  "case-sensitive": "false",
  "version": 0,
  "roots": [
    {
      "external-contents": "DIR/real/A.modulemap",
      "name": "DIR/frameworks/A.framework/Modules/module.modulemap",
      "type": "file"
    },
    {
      "external-contents": "DIR/real/A.private.modulemap",
      "name": "DIR/frameworks/A.framework/Modules/module.private.modulemap",
      "type": "file"
    },
  ]
}

//--- cdb.json.template
[
{
  "file": "DIR/tu1.m",
  "directory": "DIR",
  "command": "clang -fmodules -fmodules-cache-path=DIR/cache -fmodule-name=A -ivfsoverlay DIR/overlay.json -F DIR/frameworks -fsyntax-only DIR/tu1.m"
},
{
  "file": "DIR/tu2.m",
  "directory": "DIR",
  "command": "clang -fmodules -fmodules-cache-path=DIR/cache -fmodule-name=A -ivfsoverlay DIR/overlay.json -F DIR/frameworks -fsyntax-only DIR/tu2.m"
},
{
  "file": "DIR/tu3.m",
  "directory": "DIR",
  "command": "clang -fmodules -fmodules-cache-path=DIR/cache -fmodule-name=A -ivfsoverlay DIR/overlay.json -F DIR/frameworks -fsyntax-only DIR/tu3.m"
}
]

//--- tu1.m
#import <A/A.h>

//--- tu2.m
#import <A/A_Private.h>

//--- tu3.m
#import <B/B.h>
// The following code triggers a note diagnostic pointing to A.h which is
// resolved relative to the module base directory, which is affected by the
// modulemap path.
struct A { float x; }; // expected-error {{incompatible definitions}} expected-note {{type 'float' here}}
// expected-note@frameworks/A.framework/Headers/A.h:1 {{type 'int' here}}

// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: sed -e "s|DIR|%/t|g" %t/overlay.json.template > %t/overlay.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full -j 1 > %t/result.json
// RUN: cat %t/result.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t
// CHECK:      {
// CHECK:        "modules": [
// CHECK:          {
// CHECK:            "clang-module-deps": []
// CHECK:            "command-line": [
// CHECK:              "-x"
// CHECK-NEXT:         "objective-c"
// CHECK-NEXT:         "[[PREFIX]]/frameworks/A.framework/Modules/module.modulemap"
// CHECK:            ]
// CHECK:            "name": "A"
// CHECK:          }
// CHECK:          {
// CHECK:            "clang-module-deps": [
// CHECK:              {
// CHECK:                "module-name": "A"
// CHECK:              }
// CHECK:            ]
// CHECK:            "command-line": [
// CHECK:              "-fmodule-map-file=[[PREFIX]]/frameworks/A.framework/Modules/module.modulemap",
// CHECK:              "-x"
// CHECK-NEXT:         "objective-c"
// CHECK-NEXT:         "[[PREFIX]]/frameworks/B.framework/Modules/module.modulemap"
// CHECK:            ]
// CHECK:            "name": "B"
// CHECK:          }

// CHECK:        "translation-units": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "commands": [
// CHECK:              {
// CHECK:                "command-line": [
// CHECK:                  "-fmodule-map-file=[[PREFIX]]/frameworks/A.framework/Modules/module.modulemap"
// CHECK:                  "-fmodule-name=A"
// CHECK:                ],
// CHECK:                "input-file": "[[PREFIX]]/tu1.m"
// CHECK-NEXT:         }
// CHECK:            ]
// CHECK:          }
// CHECK-NEXT:     {
// CHECK-NEXT:       "commands": [
// CHECK:              {
// CHECK:                "command-line": [
// CHECK:                  "-fmodule-map-file=[[PREFIX]]/frameworks/A.framework/Modules/module.modulemap"
// CHECK:                  "-fmodule-name=A"
// CHECK:                ],
// CHECK:                "input-file": "[[PREFIX]]/tu2.m"
// CHECK-NEXT:         }
// CHECK:            ]
// CHECK:          }
// CHECK-NEXT:     {
// CHECK-NEXT:       "commands": [
// CHECK:              {
// CHECK:                "clang-module-deps": [
// CHECK:                  {
// CHECK:                    "module-name": "B"
// CHECK:                  }
// CHECK:                ]
// CHECK:                "command-line": [
// CHECK:                  "-fmodule-map-file=[[PREFIX]]/frameworks/A.framework/Modules/module.modulemap"
// CHECK:                  "-fmodule-map-file=[[PREFIX]]/frameworks/B.framework/Modules/module.modulemap"
// CHECK:                  "-fmodule-name=A"
// CHECK:                ],
// CHECK:                "input-file": "[[PREFIX]]/tu3.m"
// CHECK-NEXT:         }
// CHECK:            ]
// CHECK:          }

// RUN: %deps-to-rsp %t/result.json --tu-index=0 > %t/tu1.rsp
// RUN: %clang @%t/tu1.rsp

// RUN: %deps-to-rsp %t/result.json --tu-index=1 > %t/tu2.rsp
// RUN: %clang @%t/tu2.rsp

// RUN: %deps-to-rsp %t/result.json --module-name A > %t/A.rsp
// RUN: %deps-to-rsp %t/result.json --module-name B > %t/B.rsp
// RUN: %deps-to-rsp %t/result.json --tu-index=2 > %t/tu3.rsp
// RUN: %clang @%t/A.rsp
// RUN: %clang @%t/B.rsp
// RUN: %clang @%t/tu3.rsp -verify
