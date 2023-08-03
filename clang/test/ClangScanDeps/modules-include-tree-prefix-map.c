// REQUIRES: ondisk_cas

// RUN: rm -rf %t
// RUN: split-file %s %t/dir1
// RUN: cp -r %t/dir1 %t/dir2
// RUN: sed -e "s|DIR|%/t/dir1|g" -e "s|CLANG|%clang|g" %t/dir1/cdb.json.template > %t/cdb1.json
// RUN: sed -e "s|DIR|%/t/dir2|g" -e "s|CLANG|%clang|g" %t/dir1/cdb.json.template > %t/cdb2.json

// RUN: clang-scan-deps -compilation-database %t/cdb1.json \
// RUN:   -cas-path %t/cas -module-files-dir %t/dir1/outputs \
// RUN:   -prefix-map=%t/dir1/outputs=/^modules -prefix-map=%t/dir1=/^src -prefix-map-sdk=/^sdk -prefix-map-toolchain=/^tc \
// RUN:   -format experimental-include-tree-full -mode preprocess-dependency-directives \
// RUN:   > %t/deps.json

// Extract the include-tree commands
// RUN: %deps-to-rsp %t/deps.json --module-name Top > %t/Top.rsp
// RUN: %deps-to-rsp %t/deps.json --module-name Left > %t/Left.rsp
// RUN: %deps-to-rsp %t/deps.json --module-name Right > %t/Right.rsp
// RUN: %deps-to-rsp %t/deps.json --module-name System > %t/System.rsp
// RUN: %deps-to-rsp %t/deps.json --tu-index 0 > %t/tu.rsp

// Extract include-tree casids
// RUN: cat %t/Top.rsp | sed -E 's|.*"-fcas-include-tree" "(llvmcas://[[:xdigit:]]+)".*|\1|' > %t/Top.casid
// RUN: cat %t/Left.rsp | sed -E 's|.*"-fcas-include-tree" "(llvmcas://[[:xdigit:]]+)".*|\1|' > %t/Left.casid
// RUN: cat %t/Right.rsp | sed -E 's|.*"-fcas-include-tree" "(llvmcas://[[:xdigit:]]+)".*|\1|' > %t/Right.casid
// RUN: cat %t/System.rsp | sed -E 's|.*"-fcas-include-tree" "(llvmcas://[[:xdigit:]]+)".*|\1|' > %t/System.casid
// RUN: cat %t/tu.rsp | sed -E 's|.*"-fcas-include-tree" "(llvmcas://[[:xdigit:]]+)".*|\1|' > %t/tu.casid

// RUN: echo "MODULE Top" > %t/result.txt
// RUN: clang-cas-test -cas %t/cas -print-include-tree @%t/Top.casid >> %t/result.txt
// RUN: echo "MODULE Left" >> %t/result.txt
// RUN: clang-cas-test -cas %t/cas -print-include-tree @%t/Left.casid >> %t/result.txt
// RUN: echo "MODULE Right" >> %t/result.txt
// RUN: clang-cas-test -cas %t/cas -print-include-tree @%t/Right.casid >> %t/result.txt
// RUN: echo "TRANSLATION UNIT" >> %t/result.txt
// RUN: clang-cas-test -cas %t/cas -print-include-tree @%t/tu.casid >> %t/result.txt

// RUN: clang-cas-test -cas %t/cas -print-include-tree @%t/System.casid | grep '<module-includes>' | sed 's|.* llvmcas|llvmcas|' > %t/system-module-includes.casid
// RUN: echo "System module-includes" >> %t/result.txt
// RUN: llvm-cas -cas %t/cas -cat-blob @%t/system-module-includes.casid >> %t/result.txt

// RUN: FileCheck %s -input-file %t/result.txt -DPREFIX=%/t -check-prefix=NO_PATHS
// NO_PATHS-NOT: [[PREFIX]]

// RUN: cat %t/deps.json >> %t/result.txt

// RUN: FileCheck %s -input-file %t/result.txt -DPREFIX=%/t

// CHECK-LABEL: MODULE Top
// CHECK: <module-includes> llvmcas://{{[[:xdigit:]]+}}
// CHECK: 1:1 <built-in> llvmcas://{{[[:xdigit:]]+}}
// CHECK: 2:1 /^src/Top.h llvmcas://{{[[:xdigit:]]+}}
// CHECK: Module Map:
// CHECK: Top
// CHECK:   export *
// CHECK: Files:
// CHECK-NOT: module.modulemap
// CHECK: /^src/Top.h llvmcas://{{[[:xdigit:]]+}}
// CHECK-NOT: module.modulemap

// CHECK-LABEL: MODULE Left
// CHECK: <module-includes> llvmcas://{{[[:xdigit:]]+}}
// CHECK: 1:1 <built-in> llvmcas://{{[[:xdigit:]]+}}
// CHECK: 2:1 /^src/Left.h llvmcas://{{[[:xdigit:]]+}}
// CHECK:   2:1 (Module) Top
// CHECK: Module Map:
// CHECK: Left
// CHECK:   export *
// CHECK: Files:
// CHECK-NOT: module.modulemap
// CHECK: /^src/Left.h llvmcas://{{[[:xdigit:]]+}}
// CHECK-NOT: module.modulemap
// CHECK: /^src/Top.h llvmcas://{{[[:xdigit:]]+}}

// CHECK-LABEL: MODULE Right
// CHECK: <module-includes> llvmcas://{{[[:xdigit:]]+}}
// CHECK: 1:1 <built-in> llvmcas://{{[[:xdigit:]]+}}
// CHECK: 2:1 /^src/Right.h llvmcas://{{[[:xdigit:]]+}}
// CHECK:   2:1 (Module) Top
// CHECK: Module Map:
// CHECK: Right
// CHECK:   export *
// CHECK: Files:
// CHECK-NOT: module.modulemap
// CHECK: /^src/Right.h llvmcas://{{[[:xdigit:]]+}}
// CHECK-NOT: module.modulemap
// CHECK: /^src/Top.h llvmcas://{{[[:xdigit:]]+}}

// CHECK-LABEL: TRANSLATION UNIT
// CHECK: /^src/tu.m llvmcas://{{[[:xdigit:]]+}}
// CHECK: 1:1 <built-in> llvmcas://{{[[:xdigit:]]+}}
// CHECK: 2:1 (Module) Left
// CHECK: 3:1 (Module) Right

// Note: the modules with explicit imports are imported via parser and are not
// recorded in the include-tree; it's handled entirely by fmodule-map-file,
// fmodule-file, and fmodule-file-cache-key options.

// CHECK-NOT: Module Map
// CHECK: Files:
// CHECK-NOT: module.modulemap
// CHECK: /^src/Left.h llvmcas://{{[[:xdigit:]]+}}
// CHECK: /^src/Top.h llvmcas://{{[[:xdigit:]]+}}
// CHECK: /^src/Right.h llvmcas://{{[[:xdigit:]]+}}

// CHECK-LABEL: System module-includes
// CHECK-NEXT: #import "sys.h"
// CHECK-NEXT: #import "/^tc/{{.*}}/stdbool.h"

// CHECK-NEXT:      {
// CHECK-NEXT   "modules": [
// CHECK-NEXT     {
// CHECK:            "cas-include-tree-id": "[[LEFT_TREE:llvmcas://[[:xdigit:]]+]]"
// CHECK:            "clang-module-deps": [
// CHECK:              {
// CHECK:                "module-name": "Top"
// CHECK:              }
// CHECK-NEXT:       ]
// CHECK:            "clang-modulemap-file": "[[PREFIX]]/dir1/module.modulemap"
// CHECK:            "command-line": [
// CHECK-NEXT:         "-cc1"
// CHECK:              "-fcas-path"
// CHECK-NEXT:         "[[PREFIX]]/cas"
// CHECK-NOT: -fmodule-map-file
// CHECK:              "-o"
// CHECK-NEXT:         "[[PREFIX]]/dir1/outputs/{{.*}}/Left-{{.*}}.pcm"
// CHECK:              "-disable-free"
// CHECK:              "-fno-pch-timestamp"
// CHECK:              "-fcas-include-tree"
// CHECK-NEXT:         "[[LEFT_TREE]]"
// CHECK:              "-fcache-compile-job"
// CHECK:              "-emit-module"
// CHECK:              "-fmodule-file-cache-key"
// CHECK-NEXT:         "/^modules/{{.*}}/Top-{{.*}}.pcm"
// CHECK-NEXT:         "llvmcas://{{[[:xdigit:]]+}}"
// CHECK:              "-fmodule-file=Top=/^modules/{{.*}}/Top-{{.*}}.pcm"
// CHECK:              "-fmodules"
// CHECK:              "-fmodule-name=Left"
// CHECK:              "-fno-implicit-modules"
// CHECK:            ]
// CHECK:            "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/dir1/Left.h"
// CHECK-NEXT:         "[[PREFIX]]/dir1/module.modulemap"
// CHECK:            ]
// CHECK:            "name": "Left"
// CHECK:          }
// CHECK-NEXT:     {
// CHECK:            "cas-include-tree-id": "[[RIGHT_TREE:llvmcas://[[:xdigit:]]+]]"
// CHECK:            "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK:                "module-name": "Top"
// CHECK:              }
// CHECK-NEXT:       ]
// CHECK:            "clang-modulemap-file": "[[PREFIX]]/dir1/module.modulemap"
// CHECK:            "command-line": [
// CHECK-NEXT:         "-cc1"
// CHECK:              "-fcas-path"
// CHECK-NEXT:         "[[PREFIX]]/cas"
// CHECK-NOT: -fmodule-map-file
// CHECK:              "-o"
// CHECK-NEXT:         "[[PREFIX]]/dir1/outputs/{{.*}}/Right-{{.*}}.pcm"
// CHECK:              "-disable-free"
// CHECK:              "-fno-pch-timestamp"
// CHECK:              "-fcas-include-tree"
// CHECK-NEXT:         "[[RIGHT_TREE]]"
// CHECK:              "-fcache-compile-job"
// CHECK:              "-emit-module"
// CHECK:              "-fmodule-file-cache-key
// CHECK-NEXT:         "/^modules/{{.*}}/Top-{{.*}}.pcm"
// CHECK-NEXT:         "llvmcas://{{[[:xdigit:]]+}}"
// CHECK:              "-fmodule-file=Top=/^modules/{{.*}}/Top-{{.*}}.pcm"
// CHECK:              "-fmodules"
// CHECK:              "-fmodule-name=Right"
// CHECK:              "-fno-implicit-modules"
// CHECK:            ]
// CHECK:            "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/dir1/Right.h"
// CHECK-NEXT:         "[[PREFIX]]/dir1/module.modulemap"
// CHECK-NEXT:       ]
// CHECK:            "name": "Right"
// CHECK:          }
// CHECK-NEXT:     {
// CHECK:            "cas-include-tree-id": "[[SYS_TREE:llvmcas://[[:xdigit:]]+]]"
// CHECK:            "clang-module-deps": []
// CHECK:            "clang-modulemap-file": "[[PREFIX]]/dir1/System/module.modulemap"
// CHECK:            "command-line": [
// CHECK-NEXT:         "-cc1"
// CHECK:              "-fcas-path"
// CHECK-NEXT:         "[[PREFIX]]/cas"
// CHECK:              "-o"
// CHECK-NEXT:         "[[PREFIX]]/dir1/outputs/{{.*}}/System-{{.*}}.pcm"
// CHECK:              "-disable-free"
// CHECK:              "-fno-pch-timestamp"
// CHECK:              "-fcas-include-tree"
// CHECK-NEXT:         "[[SYS_TREE]]"
// CHECK:              "-fcache-compile-job"
// CHECK:              "-emit-module"
// CHECK:              "-fmodules"
// CHECK:              "-fmodule-name=System"
// CHECK:              "-fno-implicit-modules"
// CHECK:            ]
// CHECK:            "file-deps": [
// CHECK-DAG:         "{{.*}}/stdbool.h"
// CHECK-DAG:         "[[PREFIX]]/dir1/System/module.modulemap"
// CHECK-DAG:         "[[PREFIX]]/dir1/System/sys.h"
// CHECK:            ]
// CHECK:            "name": "System"
// CHECK:          }
// CHECK-NEXT:     {
// CHECK:            "cas-include-tree-id": "[[TOP_TREE:llvmcas://[[:xdigit:]]+]]"
// CHECK:            "clang-module-deps": []
// CHECK:            "clang-modulemap-file": "[[PREFIX]]/dir1/module.modulemap"
// CHECK:            "command-line": [
// CHECK-NEXT:         "-cc1"
// CHECK:              "-fcas-path"
// CHECK-NEXT:         "[[PREFIX]]/cas"
// CHECK:              "-o"
// CHECK-NEXT:         "[[PREFIX]]/dir1/outputs/{{.*}}/Top-{{.*}}.pcm"
// CHECK:              "-disable-free"
// CHECK:              "-fno-pch-timestamp"
// CHECK:              "-fcas-include-tree"
// CHECK-NEXT:         "[[TOP_TREE]]"
// CHECK:              "-fcache-compile-job"
// CHECK:              "-emit-module"
// CHECK:              "-fmodules"
// CHECK:              "-fmodule-name=Top"
// CHECK:              "-fno-implicit-modules"
// CHECK:            ]
// CHECK:            "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/dir1/Top.h"
// CHECK-NEXT:         "[[PREFIX]]/dir1/module.modulemap"
// CHECK-NEXT:       ]
// CHECK:            "name": "Top"
// CHECK:          }
// CHECK:        ]
// CHECK-NEXT:   "translation-units": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "commands": [
// CHECK-NEXT:         {
// CHECK:                "cas-include-tree-id": "[[TU_TREE:llvmcas://[[:xdigit:]]+]]"
// CHECK:                "clang-module-deps": [
// CHECK-NEXT:             {
// CHECK:                    "module-name": "Left"
// CHECK:                  }
// CHECK-NEXT:             {
// CHECK:                    "module-name": "Right"
// CHECK:                  }
// CHECK-NEXT:             {
// CHECK:                    "module-name": "System"
// CHECK:                  }
// CHECK-NEXT:           ]
// CHECK:                "command-line": [
// CHECK-NEXT:             "-cc1"
// CHECK:                  "-fcas-path"
// CHECK-NEXT:             "[[PREFIX]]/cas"
// CHECK-NOT: -fmodule-map-file
// CHECK:                  "-disable-free"
// CHECK:                  "-fcas-include-tree"
// CHECK-NEXT:             "[[TU_TREE]]"
// CHECK:                  "-fcache-compile-job"
// CHECK:                  "-fsyntax-only"
// CHECK:                  "-fmodule-file-cache-key"
// CHECK-NEXT:             "/^modules/{{.*}}/Left-{{.*}}.pcm"
// CHECK-NEXT:             "llvmcas://{{[[:xdigit:]]+}}"
// CHECK:                  "-fmodule-file-cache-key"
// CHECK-NEXT:             "/^modules/{{.*}}/Right-{{.*}}.pcm"
// CHECK-NEXT:             "llvmcas://{{[[:xdigit:]]+}}"
// CHECK:                  "-fmodule-file=Left=/^modules/{{.*}}/Left-{{.*}}.pcm"
// CHECK:                  "-fmodule-file=Right=/^modules/{{.*}}/Right-{{.*}}.pcm"
// CHECK:                  "-fmodules"
// CHECK:                  "-fno-implicit-modules"
// CHECK:                ]
// CHECK:                "file-deps": [
// CHECK-NEXT:             "[[PREFIX]]/dir1/tu.m"
// CHECK-NEXT:           ]
// CHECK:                "input-file": "[[PREFIX]]/dir1/tu.m"
// CHECK:              }
// CHECK-NEXT:       ]
// CHECK-NEXT:     }
// CHECK-NEXT:   ]
// CHECK-NEXT: }

// Build the include-tree commands
// RUN: %clang @%t/Top.rsp 2>&1 | FileCheck %s -check-prefix=CACHE_MISS
// RUN: %clang @%t/Left.rsp 2>&1 | FileCheck %s -check-prefix=CACHE_MISS
// RUN: %clang @%t/Right.rsp 2>&1 | FileCheck %s -check-prefix=CACHE_MISS
// RUN: %clang @%t/System.rsp 2>&1 | FileCheck %s -check-prefix=CACHE_MISS
// RUN: %clang @%t/tu.rsp 2>&1 | FileCheck %s -check-prefix=CACHE_MISS

// Scan in a different directory
// RUN: clang-scan-deps -compilation-database %t/cdb2.json \
// RUN:   -cas-path %t/cas -module-files-dir %t/dir2/outputs \
// RUN:   -prefix-map=%t/dir2/outputs=/^modules -prefix-map=%t/dir2=/^src -prefix-map-sdk=/^sdk -prefix-map-toolchain=/^tc \
// RUN:   -format experimental-include-tree-full -mode preprocess-dependency-directives \
// RUN:   > %t/deps2.json

// RUN: %deps-to-rsp %t/deps2.json --module-name Top > %t/Top2.rsp
// RUN: %deps-to-rsp %t/deps2.json --module-name Left > %t/Left2.rsp
// RUN: %deps-to-rsp %t/deps2.json --module-name Right > %t/Right2.rsp
// RUN: %deps-to-rsp %t/deps2.json --module-name System > %t/System2.rsp
// RUN: %deps-to-rsp %t/deps2.json --tu-index 0 > %t/tu2.rsp

// Check cache hits
// RUN: %clang @%t/Top2.rsp 2>&1 | FileCheck %s -check-prefix=CACHE_HIT
// RUN: %clang @%t/Left2.rsp 2>&1 | FileCheck %s -check-prefix=CACHE_HIT
// RUN: %clang @%t/Right2.rsp 2>&1 | FileCheck %s -check-prefix=CACHE_HIT
// RUN: %clang @%t/System2.rsp 2>&1 | FileCheck %s -check-prefix=CACHE_HIT
// RUN: %clang @%t/tu2.rsp 2>&1 | FileCheck %s -check-prefix=CACHE_HIT

// CACHE_MISS: compile job cache miss
// CACHE_HIT: compile job cache hit

//--- cdb.json.template
[{
  "file": "DIR/tu.m",
  "directory": "DIR",
  "command": "CLANG -fsyntax-only DIR/tu.m -I DIR -isystem DIR/System -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/module-cache -Rcompile-job-cache"
}]

//--- module.modulemap
module Top { header "Top.h" export *}
module Left { header "Left.h" export *}
module Right { header "Right.h" export *}

//--- Top.h
#pragma once
struct Top {
  int x;
};
void top(void);

//--- Left.h
#include "Top.h"
void left(void);

//--- Right.h
#include "Top.h"
void right(void);

//--- System/module.modulemap
module System [system] {
  header "sys.h"
  header "stdbool.h"
}

//--- System/sys.h
#include <stdbool.h>
bool sys(void);

//--- tu.m
#import "Left.h"
#import <Right.h>
#import <sys.h>

void tu(void) {
  top();
  left();
  right();
  bool b = sys();
  (void)b;
}
