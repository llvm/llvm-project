// RUN: rm -rf %t
// RUN: split-file %s %t

//--- cdb.json.template
[{
  "file": "DIR/tu.m",
  "directory": "DIR",
  "command": "clang -fmodules -fimplicit-module-maps -fmodules-cache-path=DIR/cache -fmodule-name=FWPrivate -c DIR/tu.m -o DIR/tu.o -F DIR/frameworks -Wprivate-module"
}]

//--- frameworks/FW.framework/Modules/module.modulemap
framework module FW {}
//--- frameworks/FW.framework/Modules/module.private.modulemap
// The module name will trigger a diagnostic.
framework module FWPrivate { header "private.h" }
//--- frameworks/FW.framework/PrivateHeaders/private.h
//--- tu.m

// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full > %t/result.json
// RUN: cat %t/result.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t
// CHECK:      "translation-units": [
// CHECK-NEXT:   {
// CHECK-NEXT:     "commands": [
// CHECK:            {
// CHECK:              "command-line": [
// CHECK:                "-fmodule-map-file=[[PREFIX]]/frameworks/FW.framework/Modules/module.private.modulemap",
// CHECK:                "-fmodule-name=FWPrivate",
// CHECK:              ],
// CHECK:              "file-deps": [
// CHECK-NEXT:           "[[PREFIX]]/tu.m"
// CHECK-NEXT:         ],
// CHECK-NEXT:         "input-file": "[[PREFIX]]/tu.m"
// CHECK-NEXT:       }
// CHECK:          ]
// CHECK:        }
// CHECK:      ]
