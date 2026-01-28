// RUN: rm -rf %t
// RUN: split-file %s %t

// This test interleaves valid by-name lookups with invalid by-name lookups
// to check that the valid lookups return correct results, and the invalid
// ones generate correct diagnostics.

//--- module.modulemap
module root { header "root.h" }
module root2 { header "root2.h" }
//--- root.h

//--- root2.h

//--- cdb.json.template
[{
  "file": "",
  "directory": "DIR",
  "command": "clang -fmodules -fmodules-cache-path=DIR/cache -I DIR -x c"
}]

// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: not clang-scan-deps -compilation-database %t/cdb.json -format \
// RUN:   experimental-full -module-names=modA,root,modB,modC,root2 2>&1 | \
// RUN:   FileCheck %s -DPREFIX=%/t

// CHECK: Error while scanning dependencies for modA:
// CHECK-NEXT: {{.*}}: fatal error: module 'modA' not found
// CHECK-NEXT: Error while scanning dependencies for modB:
// CHECK-NEXT: {{.*}}: fatal error: module 'modB' not found
// CHECK-NEXT: Error while scanning dependencies for modC:
// CHECK-NEXT: {{.*}}: fatal error: module 'modC' not found
// CHECK-NEXT: {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK:            "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/module.modulemap",
// CHECK-NEXT:         "[[PREFIX]]/root.h"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "link-libraries": [],
// CHECK-NEXT:       "name": "root"
// CHECK-NEXT:     },
// CHECK-NEXT:    {
// CHECK:            "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/module.modulemap",
// CHECK-NEXT:         "[[PREFIX]]/root2.h"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "link-libraries": [],
// CHECK-NEXT:       "name": "root2"
// CHECK-NEXT:     }
