// Test path prefix-mapping when using a cas-fs with clang-scan-deps in
// modules with a PCH.

// REQUIRES: ondisk_cas

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" -e "s|CLANG|%/ncclang|g" -e "s|SDK|%/S/Inputs/SDK|g" %t/cdb.json.template > %t/cdb.json
// RUN: sed -e "s|DIR|%/t|g" -e "s|CLANG|%/ncclang|g" -e "s|SDK|%/S/Inputs/SDK|g" %t/cdb_pch.json.template > %t/cdb_pch.json

// == Scan PCH
// RUN: clang-scan-deps -compilation-database %t/cdb_pch.json -format experimental-full \
// RUN:    -cas-path %t/cas -module-files-dir %t/modules \
// RUN:    -prefix-map=%t/modules=%/root^modules -prefix-map=%t=%/root^src -prefix-map-sdk=%/root^sdk -prefix-map-toolchain=%/root^tc \
// RUN:  > %t/pch_result.txt

// == Check specifics of the PCH command-line
// RUN: cat %t/pch_result.txt | %PathSanitizingFileCheck --sanitize PREFIX=%/t --sanitize SDK_PREFIX=%/S/Inputs/SDK --sanitize ROOT^=%/root^ --enable-yaml-compatibility %s -check-prefix=PCH

// == Build PCH
// RUN: %deps-to-rsp %t/pch_result.txt --module-name=B > %t/B.cc1.rsp
// RUN: %deps-to-rsp %t/pch_result.txt --module-name=A > %t/A.cc1.rsp
// RUN: %deps-to-rsp %t/pch_result.txt --tu-index 0 > %t/pch.cc1.rsp
// RUN: %clang @%t/B.cc1.rsp
// RUN: %clang @%t/A.cc1.rsp
// Ensure we load pcms from action cache
// RUN: rm -rf %t/modules
// RUN: %clang @%t/pch.cc1.rsp

// == Scan TU, including PCH
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full \
// RUN:    -cas-path %t/cas -module-files-dir %t/modules \
// RUN:    -prefix-map=%t/modules=%/root^modules -prefix-map=%t=%/root^src -prefix-map-sdk=%/root^sdk -prefix-map-toolchain=%/root^tc \
// RUN:  > %t/result.txt

// == Check specifics of the TU command-line
// RUN: cat %t/result.txt | %PathSanitizingFileCheck --sanitize PREFIX=%/t --sanitize SDK_PREFIX=%/S/Inputs/SDK --sanitize ROOT^=%/root^ --enable-yaml-compatibility %s

// == Build TU
// RUN: %deps-to-rsp %t/result.txt --module-name=C > %t/C.cc1.rsp
// RUN: %deps-to-rsp %t/result.txt --tu-index 0 > %t/tu.cc1.rsp
// RUN: %clang @%t/C.cc1.rsp
// RUN: %clang @%t/tu.cc1.rsp

// == Check the casfs.
// RUN: cat %t/A.cc1.rsp | sed -E 's/.* "-fcas-fs" "([^ ]+)" .*/\1/' > %t/A_id.txt
// RUN: cat %t/B.cc1.rsp | sed -E 's/.* "-fcas-fs" "([^ ]+)" .*/\1/' > %t/B_id.txt
// RUN: cat %t/C.cc1.rsp | sed -E 's/.* "-fcas-fs" "([^ ]+)" .*/\1/' > %t/C_id.txt
// RUN: cat %t/pch.cc1.rsp | sed -E 's/.* "-fcas-fs" "([^ ]+)" .*/\1/' > %t/pch_id.txt
// RUN: cat %t/tu.cc1.rsp | sed -E 's/.* "-fcas-fs" "([^ ]+)" .*/\1/' > %t/tu_id.txt

// RUN: llvm-cas -cas %t/cas -ls-tree-recursive @%t/A_id.txt > %t/A_fs.txt
// RUN: llvm-cas -cas %t/cas -ls-tree-recursive @%t/B_id.txt > %t/B_fs.txt
// RUN: llvm-cas -cas %t/cas -ls-tree-recursive @%t/C_id.txt > %t/C_fs.txt
// RUN: llvm-cas -cas %t/cas -ls-tree-recursive @%t/pch_id.txt > %t/pch_fs.txt
// RUN: llvm-cas -cas %t/cas -ls-tree-recursive @%t/tu_id.txt > %t/tu_fs.txt

// RUN: FileCheck %s -input-file %t/A_fs.txt -DPREFIX=%t -DSDK_PREFIX=%S/Inputs/SDK -DROOT=%{/roott} -check-prefixes=FS_NEG,FS
// RUN: FileCheck %s -input-file %t/B_fs.txt -DPREFIX=%t -DSDK_PREFIX=%S/Inputs/SDK -DROOT=%{/roott} -check-prefixes=FS_NEG,FS
// RUN: FileCheck %s -input-file %t/C_fs.txt -DPREFIX=%t -DSDK_PREFIX=%S/Inputs/SDK -DROOT=%{/roott} -check-prefixes=FS_NEG,FS
// RUN: FileCheck %s -input-file %t/pch_fs.txt -DPREFIX=%t -DSDK_PREFIX=%S/Inputs/SDK -DROOT=%{/roott} -check-prefixes=FS_NEG,FS
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

// PCH:      {
// PCH:        "modules": [
// PCH:          {
// PCH:            "casfs-root-id": "[[A_ROOT_ID:llvmcas://[[:xdigit:]]+]]"
// PCH:            "clang-module-deps": [
// PCH:              {
// PCH:                "module-name": "B"
// PCH:              }
// PCH:            ]
// PCH:            "clang-modulemap-file": "PREFIX{{/|\\\\}}module.modulemap"
// PCH:            "command-line": [
// PCH:              "-fcas-path"
// PCH:              "PREFIX{{/|\\\\}}cas"
// PCH:              "-fcas-fs"
// PCH:              "[[A_ROOT_ID]]"
// PCH:              "-fcas-fs-working-directory"
// PCH:              "ROOT^src"
// PCH:              "-fmodule-map-file=ROOT^src{{/|\\\\}}module.modulemap"
// PCH:              "-o"
// PCH:              "PREFIX{{/|\\\\}}modules{{/|\\\\}}{{.*}}{{/|\\\\}}A-{{.*}}.pcm"
// PCH:              "-fmodule-file-cache-key"
// PCH:              "ROOT^modules{{/|\\\\}}{{.*}}{{/|\\\\}}B-[[B_CONTEXT_HASH:[^.]+]].pcm"
// PCH:              "llvmcas://{{.*}}"
// PCH:              "-x"
// PCH:              "c"
// PCH:              "ROOT^src{{/|\\\\}}module.modulemap"
// PCH:              "-isysroot"
// PCH:              "ROOT^sdk"
// PCH:              "-resource-dir"
// PCH:              "ROOT^tc{{/|\\\\}}lib{{/|\\\\}}clang{{/|\\\\}}{{.*}}"
// PCH:              "-fmodule-file=B=ROOT^modules{{/|\\\\}}{{.*}}{{/|\\\\}}B-[[B_CONTEXT_HASH]].pcm"
// PCH:              "-isystem"
// PCH:              "ROOT^tc{{/|\\\\}}lib{{/|\\\\}}clang{{/|\\\\}}{{.*}}{{/|\\\\}}include"
// PCH:              "-internal-externc-isystem"
// PCH:              "ROOT^sdk{{/|\\\\}}usr{{/|\\\\}}include"
// PCH:            ]
// PCH:            "file-deps": [
// PCH:              "PREFIX{{/|\\\\}}module.modulemap"
// PCH:              "PREFIX{{/|\\\\}}a.h"
// PCH:            ]
// PCH:            "name": "A"
// PCH:          }
// PCH:          {
// PCH:            "casfs-root-id": "[[B_ROOT_ID:llvmcas://[[:xdigit:]]+]]"
// PCH:            "clang-module-deps": [],
// PCH:            "clang-modulemap-file": "PREFIX{{/|\\\\}}module.modulemap"
// PCH:            "command-line": [
// PCH:              "-fcas-path"
// PCH:              "PREFIX{{/|\\\\}}cas"
// PCH:              "-fcas-fs"
// PCH:              "[[B_ROOT_ID]]"
// PCH:              "-fcas-fs-working-directory"
// PCH:              "ROOT^src"
// PCH:              "-o"
// PCH:              "PREFIX{{/|\\\\}}modules{{/|\\\\}}{{.*}}{{/|\\\\}}B-[[B_CONTEXT_HASH]].pcm"
// PCH:              "-x"
// PCH:              "c"
// PCH:              "ROOT^src{{/|\\\\}}module.modulemap"
// PCH:              "-isysroot"
// PCH:              "ROOT^sdk"
// PCH:              "-resource-dir"
// PCH:              "ROOT^tc{{/|\\\\}}lib{{/|\\\\}}clang{{/|\\\\}}{{.*}}"
// PCH:              "-isystem"
// PCH:              "ROOT^tc{{/|\\\\}}lib{{/|\\\\}}clang{{/|\\\\}}{{.*}}{{/|\\\\}}include"
// PCH:              "-internal-externc-isystem"
// PCH:              "ROOT^sdk{{/|\\\\}}usr{{/|\\\\}}include"
// PCH:            ]
// PCH:            "context-hash": "[[B_CONTEXT_HASH]]"
// PCH:            "file-deps": [
// Note: PREFIX, SDK_PREFIX and toolchain path are unordered
// PCH-DAG:          "PREFIX{{/|\\\\}}module.modulemap"
// PCH-DAG:          "PREFIX{{/|\\\\}}b.h"
// PCH-DAG:          "SDK_PREFIX{{/|\\\\}}usr{{/|\\\\}}include{{/|\\\\}}stdlib.h"
// PCH-DAG:          "{{.*}}{{/|\\\\}}include{{/|\\\\}}stdarg.h"
// PCH:            ]
// PCH:            "name": "B"
// PCH:          }
// PCH:        ]
// PCH:        "translation-units": [
// PCH:          {
// PCH:            "commands": [
// PCH:              {
// PCH:                "casfs-root-id": "[[TU_ROOT_ID:llvmcas://[[:xdigit:]]+]]"
// PCH:                "clang-module-deps": [
// PCH:                  {
// PCH:                    "module-name": "A"
// PCH:                  }
// PCH:                ]
// PCH:                "command-line": [
// PCH:                  "-fcas-path"
// PCH:                  "PREFIX{{/|\\\\}}cas"
// PCH:                  "-fcas-fs"
// PCH:                  "[[TU_ROOT_ID]]"
// PCH:                  "-fcas-fs-working-directory"
// PCH:                  "ROOT^src"
// PCH:                  "-fmodule-map-file=ROOT^src{{/|\\\\}}module.modulemap"
// PCH:                  "-fmodule-file-cache-key"
// PCH:                  "ROOT^modules{{/|\\\\}}{{.*}}A-{{.*}}.pcm"
// PCH:                  "llvmcas://{{.*}}"
// PCH:                  "-x"
// PCH:                  "c-header"
// PCH:                  "ROOT^src{{/|\\\\}}prefix.h"
// PCH:                  "-isysroot"
// PCH:                  "ROOT^sdk"
// PCH:                  "-resource-dir"
// PCH:                  "ROOT^tc{{/|\\\\}}lib{{/|\\\\}}clang{{/|\\\\}}{{.*}}"
// PCH:                  "-fmodule-file=A=ROOT^modules{{/|\\\\}}{{.*}}{{/|\\\\}}A-{{.*}}.pcm"
// PCH:                  "-isystem"
// PCH:                  "ROOT^tc{{/|\\\\}}lib{{/|\\\\}}clang{{/|\\\\}}{{.*}}{{/|\\\\}}include"
// PCH:                  "-internal-externc-isystem"
// PCH:                  "ROOT^sdk{{/|\\\\}}usr{{/|\\\\}}include"
// PCH:                ],
// PCH:                "file-deps": [
// PCH:                  "PREFIX{{/|\\\\}}prefix.h"
// PCH:                ]
// PCH:                "input-file": "PREFIX{{/|\\\\}}prefix.h"
// PCH:              }

// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK:            "casfs-root-id": "[[C_ROOT_ID:llvmcas://[[:xdigit:]]+]]"
// CHECK:            "clang-module-deps": []
// CHECK:            "command-line": [
// CHECK-NEXT:         "-cc1"
// CHECK:              "-fcas-path"
// CHECK-NEXT:         "PREFIX{{.}}cas"
// CHECK:              "-fcas-fs"
// CHECK-NEXT:         "[[C_ROOT_ID]]"
// CHECK:              "-fcas-fs-working-directory"
// CHECK-NEXT:         "ROOT^src"
// CHECK:              "-fcache-compile-job"
// CHECK:              "-emit-module"
// CHECK:              "-fmodule-file=ROOT^modules{{/|\\\\}}{{.*}}{{/|\\\\}}B-{{.*}}.pcm"
// CHECK:              "-fmodule-file-cache-key"
// CHECK:              "ROOT^modules{{/|\\\\}}{{.*}}{{/|\\\\}}B-{{.*}}.pcm"
// CHECK:              "llvmcas://{{.*}}"
// CHECK:              "-x"
// CHECK-NEXT:         "c"
// CHECK-NEXT:         "ROOT^src{{/|\\\\}}module.modulemap"
// CHECK:              "-isysroot"
// CHECK-NEXT:         "ROOT^sdk"
// CHECK:              "-resource-dir"
// CHECK-NEXT:         "ROOT^tc{{/|\\\\}}lib{{/|\\\\}}clang{{/|\\\\}}{{.*}}"
// CHECK-NOT: PREFIX
// CHECK-NOT: SDK_PREFIX
// CHECK:            ]
// CHECK:            "name": "C"
// CHECK:          }
// CHECK-NEXT:   ]
// CHECK:        "translation-units": [
// CHECK:          {
// CHECK:            "commands": [
// CHECK:              {
// CHECK:                "casfs-root-id": "[[TU_ROOT_ID:llvmcas://[[:xdigit:]]+]]"
// CHECK:                "clang-module-deps": [
// CHECK:                  {
// CHECK:                    "module-name": "C"
// CHECK:                  }
// CHECK:                ]
// CHECK:                "command-line": [
// CHECK:                  "-fcas-path"
// CHECK-NEXT:             "PREFIX{{/|\\\\}}cas"
// CHECK:                  "-fcas-fs"
// CHECK-NEXT:             "[[TU_ROOT_ID]]"
// CHECK:                  "-fcas-fs-working-directory"
// CHECK-NEXT:             "ROOT^src"
// CHECK:                  "-fmodule-map-file=ROOT^src{{/|\\\\}}module.modulemap"
// CHECK:                  "-fmodule-file-cache-key"
// CHECK:                  "ROOT^modules{{/|\\\\}}{{.*}}C-{{.*}}.pcm"
// CHECK:                  "llvmcas://{{.*}}"
// CHECK:                  "-x"
// CHECK-NEXT:             "c"
// CHECK-NEXT:             "ROOT^src{{/|\\\\}}t.c"
// CHECK:                  "-isysroot"
// CHECK-NEXT:             "ROOT^sdk"
// CHECK:                  "-resource-dir"
// CHECK-NEXT:             "ROOT^tc{{/|\\\\}}lib{{/|\\\\}}clang{{/|\\\\}}{{.*}}"
// CHECK:                  "-fmodule-file=C=ROOT^modules{{/|\\\\}}{{.*}}{{/|\\\\}}C-{{.*}}.pcm"
// CHECK:                  "-isystem"
// CHECK-NEXT:             "ROOT^sdk{{/|\\\\}}usr{{/|\\\\}}local{{/|\\\\}}include"
// CHECK:                  "-isystem"
// CHECK-NEXT:             "ROOT^tc{{/|\\\\}}lib{{/|\\\\}}clang{{/|\\\\}}{{.*}}{{/|\\\\}}include"
// CHECK:                  "-internal-externc-isystem"
// CHECK-NEXT:             "ROOT^sdk{{/|\\\\}}usr{{/|\\\\}}include"
// CHECK:                  "-include-pch"
// CHECK-NEXT:             "ROOT^src{{/|\\\\}}prefix.h.pch"
// CHECK:                ],
// CHECK:                "file-deps": [
// CHECK:                  "PREFIX{{/|\\\\}}t.c"
// CHECK:                  "PREFIX{{/|\\\\}}prefix.h.pch"
// CHECK:                ]
// CHECK:                "input-file": "PREFIX{{/|\\\\}}t.c"
// CHECK:              }

//--- cdb.json.template
[
  {
    "directory": "DIR",
    "command": "CLANG -fsyntax-only DIR/t.c -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/mcp -target x86_64-apple-macos11 -isysroot SDK -include DIR/prefix.h",
    "file": "DIR/t.c"
  }
]

//--- cdb_pch.json.template
[
  {
    "directory" : "DIR",
    "command" : "CLANG -x c-header DIR/prefix.h -o DIR/prefix.h.pch -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/mcp -target x86_64-apple-macos11 -isysroot SDK",
    "file" : "DIR/prefix.h"
  },
]

//--- t.c
#include "c.h"

//--- prefix.h
#include "a.h"

//--- module.modulemap
module A { header "a.h" }
module B { header "b.h" }
module C { header "c.h" }

//--- a.h
#include "b.h"

//--- b.h
#include <stdarg.h>
#include <stdlib.h>

//--- c.h
#include "b.h"
