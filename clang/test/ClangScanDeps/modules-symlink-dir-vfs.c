// This test checks that we're not canonicalizing framework directories that
// play a role in VFS remapping. This could lead header search to fail when
// building that module.

// RUN: rm -rf %t
// RUN: split-file %s %t

// REQUIRES: shell

// RUN: mkdir -p %t/frameworks-symlink
// RUN: ln -s %t/frameworks/FW.framework %t/frameworks-symlink/FW.framework

// RUN: mkdir -p %t/copy
// RUN: cp %t/frameworks/FW.framework/Headers/FW.h     %t/copy
// RUN: cp %t/frameworks/FW.framework/Headers/Header.h %t/copy

//--- frameworks/FW.framework/Modules/module.modulemap
framework module FW { umbrella header "FW.h" }
//--- frameworks/FW.framework/Headers/FW.h
#import <FW/Header.h>
//--- frameworks/FW.framework/Headers/Header.h
// empty

//--- tu.m
@import FW;

//--- overlay.json.template
{
  "version": 0,
  "case-sensitive": "false",
  "roots": [
    {
      "contents": [
        {
          "external-contents": "DIR/copy/Header.h",
          "name": "Header.h",
          "type": "file"
        },
        {
          "external-contents": "DIR/copy/FW.h",
          "name": "FW.h",
          "type": "file"
        }
      ],
      "name": "DIR/frameworks-symlink/FW.framework/Headers",
      "type": "directory"
    }
  ]
}

//--- cdb.json.template
[{
  "directory": "DIR",
  "file": "DIR/tu.m",
  "command": "clang -fmodules -fmodules-cache-path=DIR/cache -ivfsoverlay DIR/overlay.json -F DIR/frameworks-symlink -c DIR/tu.m -o DIR/tu.o -Werror=non-modular-include-in-framework-module"
}]

// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: sed -e "s|DIR|%/t|g" %t/overlay.json.template > %t/overlay.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full > %t/result.json
// RUN: cat %t/result.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t

// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/frameworks-symlink/FW.framework/Modules/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK-NEXT:         "-cc1",
// CHECK:              "-emit-module",
// CHECK-NEXT:         "-x",
// CHECK-NEXT:         "objective-c",
// CHECK-NEXT:         "[[PREFIX]]/frameworks-symlink/FW.framework/Modules/module.modulemap",
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "{{.*}}",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/copy/FW.h",
// CHECK-NEXT:         "[[PREFIX]]/copy/Header.h",
// CHECK-NEXT:         "[[PREFIX]]/frameworks-symlink/FW.framework/Modules/module.modulemap"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "link-libraries": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "isFramework": true,
// CHECK-NEXT:           "link-name": "FW"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "name": "FW"
// CHECK-NEXT:     }
// CHECK-NEXT:   ],
// CHECK-NEXT:   "translation-units": [
// CHECK:        ]
// CHECK:      }

// RUN: %deps-to-rsp %t/result.json --module-name=FW > %t/FW.cc1.rsp
// RUN: %clang @%t/FW.cc1.rsp
