// REQUIRES: ondisk_cas
// REQUIRES: system-darwin

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: sed "s|DIR|%/t|g" %t/cdb_pch.json.template > %t/cdb_pch.json
// RUN: sed "s|DIR|%/t|g" %t/cas-config.template > %t/.cas-config

/// Scan PCH
// RUN: clang-scan-deps -compilation-database %t/cdb_pch.json \
// RUN:   -cas-path %t/cas -module-files-dir %t/outputs \
// RUN:   -format experimental-include-tree-full -mode preprocess-dependency-directives \
// RUN:   > %t/deps_pch.json

/// Build PCH
// RUN: %deps-to-rsp %t/deps_pch.json --module-name Top > %t/Top.rsp
// RUN: %deps-to-rsp %t/deps_pch.json --module-name Left > %t/Left.rsp
// RUN: %deps-to-rsp %t/deps_pch.json --tu-index 0 > %t/pch.rsp
// RUN: %clang @%t/Top.rsp
// RUN: %clang @%t/Left.rsp
// RUN: %clang @%t/pch.rsp

/// Scan TU with PCH
// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -cas-path %t/cas -module-files-dir %t/outputs \
// RUN:   -format experimental-include-tree-full -mode preprocess-dependency-directives \
// RUN:   > %t/deps.json

// RUN: cat %t/deps.json | %PathSanitizingFileCheck --sanitize PREFIX=%/t %s

/// Build TU
// RUN: %deps-to-rsp %t/deps.json --module-name Right > %t/Right.rsp
// RUN: %deps-to-rsp %t/deps.json --tu-index 0 > %t/tu.rsp
// RUN: %clang @%t/Right.rsp
// RUN: %clang @%t/tu.rsp

/// Check module in a different directory and compare output.
// RUN: clang-scan-deps -compilation-database %t/cdb_pch.json \
// RUN:   -cas-path %t/cas -module-files-dir %t/outputs-2 \
// RUN:   -format experimental-include-tree-full -mode preprocess-dependency-directives \
// RUN:   > %t/deps_pch_2.json
// RUN: %deps-to-rsp %t/deps_pch_2.json --module-name Top > %t/Top_2.rsp
// RUN: %deps-to-rsp %t/deps_pch_2.json --module-name Left > %t/Left_2.rsp
// RUN: %deps-to-rsp %t/deps_pch_2.json --tu-index 0 > %t/pch_2.rsp
// RUN: %clang @%t/Top_2.rsp
// RUN: %clang @%t/Left_2.rsp
// RUN: %clang @%t/pch_2.rsp -o %t/prefix_2.pch
// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -cas-path %t/cas -module-files-dir %t/outputs-2 \
// RUN:   -format experimental-include-tree-full -mode preprocess-dependency-directives \
// RUN:   > %t/deps_2.json
// RUN: %deps-to-rsp %t/deps_2.json --module-name Right > %t/Right_2.rsp
// RUN: %deps-to-rsp %t/deps_2.json --tu-index 0 > %t/tu_2.rsp
// RUN: %clang @%t/Right_2.rsp
// RUN: %clang @%t/tu_2.rsp -o %t/tu_2.o

/// Diff all outputs
// RUN: diff %t/prefix.h.pch %t/prefix_2.pch
// RUN: diff %t/tu.o %t/tu_2.o


// CHECK:      {
// CHECK-NEXT:  "modules": [
// CHECK-NEXT:     {
// CHECK:            "clang-module-deps": []
// CHECK:            "clang-modulemap-file": "PREFIX{{/|\\\\}}module.modulemap"
// CHECK:            "command-line": [
// CHECK-NEXT:         "-cc1"
// CHECK:              "-fcas-path"
// CHECK-NEXT:         "PREFIX{{/|\\\\}}cas"
// CHECK:              "-o"
// CHECK-NEXT:         "PREFIX{{/|\\\\}}outputs{{/|\\\\}}{{.*}}{{/|\\\\}}Right-{{.*}}.pcm"
// CHECK:              "-disable-free"
// CHECK:              "-fno-pch-timestamp"
// CHECK:              "-fcas-include-tree"
// CHECK-NEXT:         "[[RIGHT_TREE:llvmcas://[[:xdigit:]]+]]"
// CHECK:              "-fcache-compile-job"
// CHECK:              "-emit-module"
// CHECK:              "-fmodule-file=Top-{{.*}}.pcm"
// CHECK:              "-fmodule-file-cache-key"
// CHECK-NEXT:         "Top-{{.*}}.pcm"
// CHECK-NEXT:         "llvmcas://{{[[:xdigit:]]+}}"
// CHECK:              "-x"
// CHECK-NEXT:         "c"
// CHECK:              "-fmodules"
// CHECK:              "-fmodule-name=Right"
// CHECK:              "-fno-implicit-modules"
// CHECK:            ]
// CHECK:            "file-deps": [
// CHECK-NEXT:         "PREFIX{{/|\\\\}}module.modulemap"
// CHECK-NEXT:         "PREFIX{{/|\\\\}}Right.h"
// CHECK-NEXT:       ]
// CHECK:            "name": "Right"
// CHECK:          }
// CHECK-NOT: "clang-modulemap-file"
// CHECK:        ]
// CHECK-NEXT:   "translation-units": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "commands": [
// CHECK-NEXT:         {
// CHECK:                "clang-module-deps": [
// CHECK-NEXT:             {
// CHECK:                    "module-name": "Right"
// CHECK:                  }
// CHECK-NEXT:           ]
// CHECK:                "command-line": [
// CHECK-NEXT:             "-cc1"
// CHECK:                  "-fcas-path"
// CHECK-NEXT:             "PREFIX{{/|\\\\}}cas"
// CHECK-NOT: -fmodule-map-file=
// CHECK:                  "-disable-free"
// CHECK:                  "-fcas-include-tree"
// CHECK-NEXT:             "llvmcas://{{[[:xdigit:]]+}}"
// CHECK:                  "-fcache-compile-job"
// CHECK:                  "-fmodule-file-cache-key"
// CHECK-NEXT:             "Right-{{.*}}.pcm"
// CHECK-NEXT:             "llvmcas://{{[[:xdigit:]]+}}"
// CHECK:                  "-x"
// CHECK-NEXT:             "c"
// CHECK:                  "-fmodule-file=Right=Right-{{.*}}.pcm"
// CHECK:                  "-fmodules"
// CHECK:                  "-fno-implicit-modules"
// CHECK:                ]
// CHECK:                "file-deps": [
// CHECK-NEXT:             "PREFIX{{/|\\\\}}tu.c"
// CHECK-NEXT:             "PREFIX{{/|\\\\}}prefix.h.pch"
// CHECK-NEXT:           ]
// CHECK:                "input-file": "PREFIX{{/|\\\\}}tu.c"
// CHECK:              }
// CHECK-NEXT:       ]
// CHECK-NEXT:     }
// CHECK-NEXT:   ]
// CHECK-NEXT: }

//--- cdb_pch.json.template
[{
  "file": "DIR/prefix.h",
  "directory": "DIR",
  "command": "clang -x c-header DIR/prefix.h -o DIR/prefix.h.pch -fmodules -fimplicit-modules -fimplicit-module-maps -gmodules -fmodules-cache-path=DIR/module-cache"
}]

//--- cdb.json.template
[{
  "file": "DIR/tu.c",
  "directory": "DIR",
  "command": "clang -c -o DIR/tu.o DIR/tu.c -include DIR/prefix.h -fmodules -fimplicit-modules -fimplicit-module-maps -gmodules -fmodules-cache-path=DIR/module-cache"
}]

//--- module.modulemap
module Top { header "Top.h" export *}
module Left { header "Left.h" export *}
module Right { header "Right.h" export *}

//--- Top.h
#pragma once
struct Top { int x; };

//--- Left.h
#pragma once
#include "Top.h"
struct Left { struct Top top; };

//--- Right.h
#pragma once
#include "Top.h"
struct Right { struct Top top; };

//--- prefix.h
#include "Left.h"
struct Prefix { struct Top top; };

//--- tu.c
#include "Right.h"

int main(void) {
  struct Left _left;
  struct Right _right;
  struct Top _top;
  struct Prefix _prefix;
  return 0;
}

//--- cas-config.template
{
  "CASPath": "DIR/cas"
}
