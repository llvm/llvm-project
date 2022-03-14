// RUN: rm -rf %t && mkdir %t
// RUN: cp -r %S/Inputs/preserved-args/* %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full > %t/result.json
// RUN: cat %t/result.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t

// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK:            "command-line": [
// CHECK-NEXT:         "-cc1"
// CHECK:              "-serialize-diagnostic-file"
// CHECK-NEXT:         "[[PREFIX]]/tu.dia"
// CHECK:              "-fmodule-file=Foo=[[PREFIX]]/foo.pcm"
// CHECK:              "-MT"
// CHECK-NEXT:         "my_target"
// CHECK:              "-dependency-file"
// CHECK-NEXT:         "[[PREFIX]]/tu.d"
// CHECK:            ],
// CHECK:            "name": "Mod"
// CHECK-NEXT:     }
// CHECK-NEXT:   ]
// CHECK:      }
