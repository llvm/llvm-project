// Test path prefix-mapping when using a cas-fs with clang-scan-deps in
// tree, full-tree, and full dependencies modes.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%t|g" -e "s|CLANG|%clang|g" -e "s|SDK|%S/Inputs/SDK|g" %t/cdb.json.template > %t/cdb.json

// == Tree
// Ensure the filesystem has the mapped paths.

// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-tree -cas-path %t/cas \
// RUN:    -prefix-map=%t=/^src -prefix-map-sdk=/^sdk -prefix-map-toolchain=/^tc \
// RUN:  | sed -E 's/tree ([^ ]+) for.*/\1/' > %t/tree_id.txt
// RUN: llvm-cas -cas %t/cas -ls-tree-recursive @%t/tree_id.txt > %t/tree_result.txt
// RUN: FileCheck %s -input-file %t/tree_result.txt -check-prefix=FILES

// FILES: file llvmcas://{{.*}} /^sdk/usr/include/stdlib.h
// FILES: file llvmcas://{{.*}} /^src/t.c
// FILES: file llvmcas://{{.*}} /^src/top.h
// FILES: file llvmcas://{{.*}} /^tc/lib/clang/{{.*}}/include/stdarg.h

// == Full Tree
// This should have the same filesystem as above, and we also check the command-
// line.

// RUN: cat %t/tree_id.txt > %t/full_tree_result.txt
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-tree-full -cas-path %t/cas \
// RUN:    -prefix-map=%t=/^src -prefix-map-sdk=/^sdk -prefix-map-toolchain=/^tc \
// RUN:  >> %t/full_tree_result.txt
// RUN: FileCheck %s -input-file %t/full_tree_result.txt -DPREFIX=%t -DSDK_PREFIX=%S/Inputs/SDK

// == Full
// Same as full tree.

// RUN: cat %t/tree_id.txt > %t/full_result.txt
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full -cas-path %t/cas \
// RUN:    -prefix-map=%t=/^src -prefix-map-sdk=/^sdk -prefix-map-toolchain=/^tc \
// RUN:  >> %t/full_result.txt
// RUN: FileCheck %s -input-file %t/full_result.txt -DPREFIX=%t -DSDK_PREFIX=%S/Inputs/SDK

// CHECK: [[MAPPED_FS_ID:llvmcas://[[:xdigit:]]+]]
// CHECK:      "modules": []
// CHECK:      "translation-units": [
// CHECK:        {
// CHECK:          "commands": [
// CHECK:            {
// CHECK:              "casfs-root-id": "[[MAPPED_FS_ID]]"
// CHECK:              "clang-module-deps": []
// CHECK:              "command-line": [
// CHECK:                "-fcas-path"
// CHECK-NEXT:           "[[PREFIX]]/cas"
// CHECK:                "-fcas-fs"
// CHECK-NEXT:           "[[MAPPED_FS_ID]]"
// CHECK:                "-fcas-fs-working-directory"
// CHECK-NEXT:           "/^src"
// CHECK:                "-x"
// CHECK-NEXT:           "c"
// CHECK-NEXT:           "/^src/t.c"
// CHECK:                "-isysroot"
// CHECK-NEXT:           "/^sdk"
// CHECK:                "-resource-dir"
// CHECK-NEXT:           "/^tc/lib/clang/{{.*}}"
// CHECK:                "-isystem"
// CHECK-NEXT:           "/^sdk/usr/local/include
// CHECK:                "-isystem"
// CHECK-NEXT:           "/^tc/lib/clang/{{.*}}/include"
// CHECK:                "-internal-externc-isystem"
// CHECK-NEXT:           "/^sdk/usr/include"
// CHECK:                "-fdebug-compilation-dir=/^src"
// CHECK:                "-fcoverage-compilation-dir=/^src"
// CHECK-NOT: [[PREFIX]]
// CHECK-NOT: [[SDK_PREFIX]]
// CHECK:              ]
// CHECK:              "file-deps": [
// CHECK:                "[[PREFIX]]/t.c"
// CHECK:                "[[PREFIX]]/top.h"
// CHECK:                "{{.*}}include/stdarg.h"
// CHECK:                "[[SDK_PREFIX]]/usr/include/stdlib.h"
// CHECK:              ]
// CHECK:              "input-file": "[[PREFIX]]/t.c"

//--- cdb.json.template
[
  {
    "directory": "DIR",
    "command": "CLANG -fsyntax-only DIR/t.c -target x86_64-apple-macos11 -isysroot SDK",
    "file": "DIR/t.c"
  }
]

//--- t.c
#include "top.h"

//--- top.h
#include <stdarg.h>
#include <stdlib.h>
