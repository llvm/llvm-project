// UNSUPPORTED: target=powerpc64-ibm-aix{{.*}}
// REQUIRES: ondisk_cas

// RUN: rm -rf %t
// RUN: split-file %s %t

//--- module.modulemap
module root { header "root.h" }
module direct { header "direct.h" }
module transitive { header "transitive.h" }
//--- root.h
#include "direct.h"
#include "root/textual.h"
//--- direct.h
#include "transitive.h"
//--- transitive.h
// empty

//--- root/textual.h
// This is here to verify that the "root" directory doesn't clash with name of
// the "root" module.

//--- cdb.json.template
[{
  "file": "",
  "directory": "DIR",
  "command": "clang -fmodules -fmodules-cache-path=DIR/cache -I DIR -x c"
}]

// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -cas-path %t/cas -format experimental-include-tree-full -module-name=root > %t/result.json
// RUN: cat %t/result.json | sed 's:\\\\\?:/:g' | FileCheck -DPREFIX=%/t %s

// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "cache-key": "[[DIRECT_CACHE_KEY:llvmcas://[[:xdigit:]]+]]"
// CHECK-NEXT:       "cas-include-tree-id": "[[LEFT_ROOT_ID:llvmcas://[[:xdigit:]]+]]"
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "{{.*}}",
// CHECK-NEXT:           "module-name": "transitive"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK:              "-fmodule-file-cache-key"
// CHECK-NEXT:         "{{.*transitive-.*\.pcm}}"
// CHECK-NEXT:         "[[TRANSITIVE_CACHE_KEY:llvmcas://[[:xdigit:]]+]]"
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "{{.*}}",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/direct.h"
// CHECK-NEXT:         "[[PREFIX]]/module.modulemap"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "name": "direct"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "cache-key": "[[ROOT_CACHE_KEY:llvmcas://[[:xdigit:]]+]]"
// CHECK-NEXT:       "cas-include-tree-id": "[[ROOT_ROOT_ID:llvmcas://[[:xdigit:]]+]]"
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "{{.*}}",
// CHECK-NEXT:           "module-name": "direct"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK:              "-fmodule-file-cache-key"
// CHECK-NEXT:         "{{.*direct-.*\.pcm}}"
// CHECK-NEXT:         "[[DIRECT_CACHE_KEY]]"
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "{{.*}}",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/module.modulemap"
// CHECK-NEXT:         "[[PREFIX]]/root.h"
// CHECK-NEXT:         "[[PREFIX]]/root/textual.h"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "name": "root"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "cache-key": "[[TRANSITIVE_CACHE_KEY]]"
// CHECK-NEXT:       "cas-include-tree-id": "[[TRANSITIVE_ROOT_ID:llvmcas://[[:xdigit:]]+]]"
// CHECK-NEXT:       "clang-module-deps": [],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "{{.*}}",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/module.modulemap"
// CHECK-NEXT:         "[[PREFIX]]/transitive.h"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "name": "transitive"
// CHECK-NEXT:     }
// CHECK-NEXT:   ],
// CHECK-NEXT:   "translation-units": []
// CHECK-NEXT: }
