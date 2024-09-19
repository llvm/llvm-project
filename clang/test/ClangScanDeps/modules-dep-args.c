// This test exercises the different ways explicit modular dependencies are
// provided on the command-line.

// RUN: rm -rf %t
// RUN: split-file %s %t

//--- cdb.json.template
[{
  "file": "DIR/tu.c",
  "directory": "DIR",
  "command": "clang DIR/tu.c -fmodules -fmodules-cache-path=DIR/cache -o DIR/tu.o"
}]

//--- module.modulemap
module Transitive { header "transitive.h" }
module Direct { header "direct.h" }
//--- transitive.h
//--- direct.h
#include "transitive.h"
//--- tu.c
#include "direct.h"

// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// Check that the PCM path defaults to the modules cache from implicit build.
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full > %t/result_cache.json
// RUN: cat %t/result_cache.json  | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t --check-prefixes=CHECK,CHECK_CACHE

// Check that the PCM path can be customized.
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full -module-files-dir %t/build > %t/result_build.json
// RUN: cat %t/result_build.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t --check-prefixes=CHECK,CHECK_BUILD

// Check that the PCM file is loaded lazily by default.
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full > %t/result_lazy.json
// RUN: cat %t/result_lazy.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t --check-prefixes=CHECK,CHECK_LAZY

// Check that the PCM file can be loaded eagerly.
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full -eager-load-pcm > %t/result_eager.json
// RUN: cat %t/result_eager.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t --check-prefixes=CHECK,CHECK_EAGER

// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "{{.*}}",
// CHECK-NEXT:           "module-name": "Transitive"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK_CACHE:        "-fmodule-file={{.*}}/cache/{{.*}}/Transitive-{{.*}}.pcm"
// CHECK_BUILD:        "-fmodule-file={{.*}}/build/{{.*}}/Transitive-{{.*}}.pcm"
// CHECK_LAZY:         "-fmodule-map-file=[[PREFIX]]/module.modulemap"
// CHECK_LAZY:         "-fmodule-file=Transitive=[[PREFIX]]/{{.*}}/Transitive-{{.*}}.pcm"
// CHECK_EAGER-NOT:    "-fmodule-map-file={{.*}}"
// CHECK_EAGER:        "-fmodule-file=[[PREFIX]]/{{.*}}/Transitive-{{.*}}.pcm"
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "{{.*}}",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/direct.h",
// CHECK-NEXT:         "[[PREFIX]]/module.modulemap"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "link-libraries": [],
// CHECK-NEXT:       "name": "Direct"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "{{.*}}",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/module.modulemap",
// CHECK-NEXT:         "[[PREFIX]]/transitive.h"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "link-libraries": [],
// CHECK-NEXT:       "name": "Transitive"
// CHECK-NEXT:     }
// CHECK-NEXT:   ],
// CHECK-NEXT:   "translation-units": [
// CHECK-NEXT:     {
// CHECK:            "clang-context-hash": "{{.*}}",
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "{{.*}}",
// CHECK-NEXT:           "module-name": "Direct"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "command-line": [
// CHECK_CACHE:        "-fmodule-file={{.*}}/cache/{{.*}}/Direct-{{.*}}.pcm"
// CHECK_BUILD:        "-fmodule-file={{.*}}/build/{{.*}}/Direct-{{.*}}.pcm"
// CHECK_LAZY:         "-fmodule-map-file=[[PREFIX]]/module.modulemap"
// CHECK_LAZY:         "-fmodule-file=Direct=[[PREFIX]]/{{.*}}/Direct-{{.*}}.pcm"
// CHECK_EAGER-NOT:    "-fmodule-map-file={{.*}}"
// CHECK_EAGER:        "-fmodule-file=[[PREFIX]]/{{.*}}/Direct-{{.*}}.pcm"
// CHECK:            ],
// CHECK:            "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/tu.c"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "input-file": "[[PREFIX]]/tu.c"
// CHECK-NEXT:     }
