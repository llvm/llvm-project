// Test path prefix-mapping when using a cas-fs with clang-scan-deps in
// modules.

// REQUIRES: ondisk_cas

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" -e "s|CLANG|%/ncclang|g" -e "s|SDK|%/S/Inputs/SDK|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full \
// RUN:    -cas-path %t/cas -module-files-dir %t/modules \
// RUN:    -prefix-map=%t/modules=%/root^modules -prefix-map=%t=%/root^src -prefix-map-sdk=%/root^sdk -prefix-map-toolchain=%/root^tc \
// RUN:  > %t/full_result.txt

// Check the command-lines.
// RUN: cat %t/full_result.txt | %PathSanitizingFileCheck --sanitize PREFIX=%/t --sanitize SDK_PREFIX=%/S/Inputs/SDK --sanitize ROOT^=%/root^ --enable-yaml-compatibility %s

// Extract individual commands.
// RUN: %deps-to-rsp %t/full_result.txt --module-name=B > %t/B.cc1.rsp
// RUN: %deps-to-rsp %t/full_result.txt --module-name=A > %t/A.cc1.rsp
// RUN: %deps-to-rsp %t/full_result.txt --tu-index 0 > %t/tu.cc1.rsp

// Check the casfs.
// RUN: cat %t/B.cc1.rsp | sed -E 's/.* "-fcas-fs" "([^ ]+)" .*/\1/' > %t/B_id.txt
// RUN: cat %t/A.cc1.rsp | sed -E 's/.* "-fcas-fs" "([^ ]+)" .*/\1/' > %t/A_id.txt
// RUN: cat %t/tu.cc1.rsp | sed -E 's/.* "-fcas-fs" "([^ ]+)" .*/\1/' > %t/tu_id.txt
// RUN: llvm-cas -cas %t/cas -ls-tree-recursive @%t/B_id.txt > %t/B_fs.txt
// RUN: llvm-cas -cas %t/cas -ls-tree-recursive @%t/A_id.txt > %t/A_fs.txt
// RUN: llvm-cas -cas %t/cas -ls-tree-recursive @%t/tu_id.txt > %t/tu_fs.txt
// RUN: FileCheck %s -input-file %t/A_fs.txt -DPREFIX=%t -DSDK_PREFIX=%S/Inputs/SDK -DROOT=%{/roott} -check-prefixes=FS_NEG,FS
// RUN: FileCheck %s -input-file %t/B_fs.txt -DPREFIX=%t -DSDK_PREFIX=%S/Inputs/SDK -DROOT=%{/roott} -check-prefixes=FS_NEG,FS
// RUN: FileCheck %s -input-file %t/tu_fs.txt -DPREFIX=%t -DSDK_PREFIX=%S/Inputs/SDK -DROOT=%{/roott} -check-prefixes=FS_NEG,FS

// FS_NEG-NOT: [[PREFIX]]
// FS_NEG-NOT: [[SDK_PREFIX]]
// FS_NEG-NOT: .pcm{{$}}
// FS: file llvmcas://{{.*}} [[ROOT]]^sdk/usr/include/stdlib.h
// FS: file llvmcas://{{.*}} [[ROOT]]^src/a.h
// FS: file llvmcas://{{.*}} [[ROOT]]^src/b.h
// FS: file llvmcas://{{.*}} [[ROOT]]^src/module.modulemap
// FS: file llvmcas://{{.*}} [[ROOT]]^tc/lib/clang/{{.*}}/include/stdarg.h

// Check that it builds.
// RUN: %clang @%t/B.cc1.rsp
// RUN: %clang @%t/A.cc1.rsp
// RUN: %clang @%t/tu.cc1.rsp

// CHECK:      {
// CHECK:        "modules": [
// CHECK:          {
// CHECK:            "casfs-root-id": "[[A_ROOT_ID:llvmcas://[[:xdigit:]]+]]"
// CHECK:            "clang-module-deps": [
// CHECK:              {
// CHECK:                "module-name": "B"
// CHECK:              }
// CHECK:            ]
// CHECK:            "clang-modulemap-file": "PREFIX{{/|\\\\}}module.modulemap"
// CHECK:            "command-line": [
// CHECK:              "-fcas-path"
// CHECK:              "PREFIX{{/|\\\\}}cas"
// CHECK:              "-fcas-fs"
// CHECK:              "[[A_ROOT_ID]]"
// CHECK:              "-fcas-fs-working-directory"
// CHECK:              "ROOT^src"
// CHECK:              "-fmodule-map-file=ROOT^src{{/|\\\\}}module.modulemap"
// CHECK:              "-o"
// CHECK:              "PREFIX{{/|\\\\}}modules{{/|\\\\}}{{.*}}{{/|\\\\}}A-{{.*}}.pcm"
// CHECK:              "-fmodule-file-cache-key"
// CHECK:              "ROOT^modules{{/|\\\\}}{{.*}}{{/|\\\\}}B-[[B_CONTEXT_HASH:[^.]+]].pcm"
// CHECK:              "llvmcas://{{.*}}"
// CHECK:              "-x"
// CHECK:              "c"
// CHECK:              "ROOT^src{{/|\\\\}}module.modulemap"
// CHECK:              "-isysroot"
// CHECK:              "ROOT^sdk"
// CHECK:              "-resource-dir"
// CHECK:              "ROOT^tc{{/|\\\\}}lib{{/|\\\\}}clang{{/|\\\\}}{{.*}}"
// CHECK:              "-fmodule-file=B=ROOT^modules{{/|\\\\}}{{.*}}{{/|\\\\}}B-[[B_CONTEXT_HASH]].pcm"
// CHECK:              "-isystem"
// CHECK:              "ROOT^tc{{/|\\\\}}lib{{/|\\\\}}clang{{/|\\\\}}{{.*}}{{/|\\\\}}include"
// CHECK:              "-internal-externc-isystem"
// CHECK:              "ROOT^sdk{{/|\\\\}}usr{{/|\\\\}}include"
// CHECK:            ]
// CHECK:            "file-deps": [
// CHECK:              "PREFIX{{/|\\\\}}module.modulemap"
// CHECK:              "PREFIX{{/|\\\\}}a.h"
// CHECK:            ]
// CHECK:            "name": "A"
// CHECK:          }
// CHECK:          {
// CHECK:            "casfs-root-id": "[[B_ROOT_ID:llvmcas://[[:xdigit:]]+]]"
// CHECK:            "clang-module-deps": [],
// CHECK:            "clang-modulemap-file": "PREFIX{{/|\\\\}}module.modulemap"
// CHECK:            "command-line": [
// CHECK:              "-fcas-path"
// CHECK:              "PREFIX{{/|\\\\}}cas"
// CHECK:              "-fcas-fs"
// CHECK:              "[[B_ROOT_ID]]"
// CHECK:              "-fcas-fs-working-directory"
// CHECK:              "ROOT^src"
// CHECK:              "-o"
// CHECK:              "PREFIX{{/|\\\\}}modules{{/|\\\\}}{{.*}}{{/|\\\\}}B-[[B_CONTEXT_HASH]].pcm"
// CHECK:              "-x"
// CHECK:              "c"
// CHECK:              "ROOT^src{{/|\\\\}}module.modulemap"
// CHECK:              "-isysroot"
// CHECK:              "ROOT^sdk"
// CHECK:              "-resource-dir"
// CHECK:              "ROOT^tc{{/|\\\\}}lib{{/|\\\\}}clang{{/|\\\\}}{{.*}}"
// CHECK:              "-isystem"
// CHECK:              "ROOT^tc{{/|\\\\}}lib{{/|\\\\}}clang{{/|\\\\}}{{.*}}{{/|\\\\}}include"
// CHECK:              "-internal-externc-isystem"
// CHECK:              "ROOT^sdk{{/|\\\\}}usr{{/|\\\\}}include"
// CHECK:            ]
// CHECK:            "context-hash": "[[B_CONTEXT_HASH]]"
// CHECK:            "file-deps": [
// Note: PREFIX, SDK_PREFIX and toolchain path are unordered
// CHECK-DAG:          "PREFIX{{/|\\\\}}module.modulemap"
// CHECK-DAG:          "PREFIX{{/|\\\\}}b.h"
// CHECK-DAG:          "{{.*}}{{/|\\\\}}include{{/|\\\\}}stdarg.h"
// CHECK-DAG:          "SDK_PREFIX{{/|\\\\}}usr{{/|\\\\}}include{{/|\\\\}}stdlib.h"
// CHECK:            ]
// CHECK:            "name": "B"
// CHECK:          }
// CHECK:        ]
// CHECK:        "translation-units": [
// CHECK:          {
// CHECK:            "commands": [
// CHECK:              {
// CHECK:                "casfs-root-id": "[[TU_ROOT_ID:llvmcas://[[:xdigit:]]+]]"
// CHECK:                "clang-module-deps": [
// CHECK:                  {
// CHECK:                    "module-name": "A"
// CHECK:                  }
// CHECK:                ]
// CHECK:                "command-line": [
// CHECK:                  "-fcas-path"
// CHECK:                  "PREFIX{{/|\\\\}}cas"
// CHECK:                  "-fcas-fs"
// CHECK:                  "[[TU_ROOT_ID]]"
// CHECK:                  "-fcas-fs-working-directory"
// CHECK:                  "ROOT^src"
// CHECK:                  "-fmodule-map-file=ROOT^src{{/|\\\\}}module.modulemap"
// CHECK:                  "-fmodule-file-cache-key"
// CHECK:                  "ROOT^modules{{/|\\\\}}{{.*}}A-{{.*}}.pcm"
// CHECK:                  "llvmcas://{{.*}}"
// CHECK:                  "-x"
// CHECK:                  "c"
// CHECK:                  "ROOT^src{{/|\\\\}}t.c"
// CHECK:                  "-isysroot"
// CHECK:                  "ROOT^sdk"
// CHECK:                  "-resource-dir"
// CHECK:                  "ROOT^tc{{/|\\\\}}lib{{/|\\\\}}clang{{/|\\\\}}{{.*}}"
// CHECK:                  "-fmodule-file=A=ROOT^modules{{/|\\\\}}{{.*}}{{/|\\\\}}A-{{.*}}.pcm"
// CHECK:                  "-isystem"
// CHECK:                  "ROOT^tc{{/|\\\\}}lib{{/|\\\\}}clang{{/|\\\\}}{{.*}}{{/|\\\\}}include"
// CHECK:                  "-internal-externc-isystem"
// CHECK:                  "ROOT^sdk{{/|\\\\}}usr{{/|\\\\}}include"
// CHECK:                ],
// CHECK:                "file-deps": [
// CHECK:                  "PREFIX{{/|\\\\}}t.c"
// CHECK:                ]
// CHECK:                "input-file": "PREFIX{{/|\\\\}}t.c"
// CHECK:              }


//--- cdb.json.template
[
  {
    "directory": "DIR",
    "command": "CLANG -fsyntax-only DIR/t.c -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/mcp -target x86_64-apple-macos11 -isysroot SDK",
    "file": "DIR/t.c"
  }
]

//--- t.c
#include "a.h"

//--- module.modulemap
module A { header "a.h" }
module B { header "b.h" }

//--- a.h
#include "b.h"

//--- b.h
#include <stdarg.h>
#include <stdlib.h>
