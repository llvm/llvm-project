// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%t|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -format experimental-include-tree-full -cas-path %t/cas \
// RUN:   -prefix-map=%t=/^src -prefix-map-sdk=/^sdk -prefix-map-toolchain=/^tc > %t/deps.json

//--- cdb.json.template
[{
  "file": "tu.c",
  "directory": "DIR",
  "command": "clang DIR/tu.c -fsyntax-only -include DIR/prefix.h"
}]

//--- prefix.h
// Note: this is a bit hacky, but we rely on the fact that dep directives lexing
// will not see #error during the scan.
#error "failed to find dependency directives"

//--- tu.c
