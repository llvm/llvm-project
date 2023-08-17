// This test checks that we report the module map that allowed inferring using
// its on-disk path in file dependencies.

// RUN: rm -rf %t
// RUN: split-file %s %t

//--- frameworks/Inferred.framework/Headers/Inferred.h
//--- frameworks/Inferred.framework/Frameworks/Sub.framework/Headers/Sub.h
//--- real/module.modulemap
framework module * {}
//--- vfsoverlay.json.template
{
  "version": 0,
  "case-sensitive": "false",
  "use-external-names": true,
  "roots": [
    {
      "contents": [
        {
           "external-contents": "DIR/real/module.modulemap",
           "name": "module.modulemap",
           "type": "file"
        }
      ],
      "name": "DIR/frameworks",
      "type": "directory"
    }
  ]
}
//--- tu.m
#include <Inferred/Inferred.h>

//--- cdb.json.template
[{
  "directory": "DIR",
  "file": "DIR/tu.m",
  "command": "clang -fmodules -fimplicit-module-maps -fmodules-cache-path=DIR/cache -F DIR/frameworks -ivfsoverlay DIR/vfsoverlay.json -c DIR/tu.m -o DIR/tu.o"
}]

// RUN: sed "s|DIR|%/t|g" %t/vfsoverlay.json.template > %t/vfsoverlay.json
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full > %t/result.json
// RUN: cat %t/result.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t

// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/frameworks/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "{{.*}}",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/frameworks/Inferred.framework/Frameworks/Sub.framework/Headers/Sub.h",
// CHECK-NEXT:         "[[PREFIX]]/frameworks/Inferred.framework/Headers/Inferred.h",
// CHECK-NEXT:         "[[PREFIX]]/real/module.modulemap"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "name": "Inferred"
// CHECK-NEXT:     }
// CHECK-NEXT:   ],
// CHECK-NEXT:   "translation-units": [
// CHECK:        ]
// CHECK:      }
