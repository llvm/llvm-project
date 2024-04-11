// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb_pch.json.template > %t/cdb_pch.json
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: sed -e "s|DIR|%/t|g" %t/cdb_no_preserve.json.template > %t/cdb_no_preserve.json

// RUN: clang-scan-deps -compilation-database %t/cdb_pch.json -format experimental-include-tree-full -cas-path %t/cas > %t/deps_pch.json
// RUN: FileCheck %s -input-file %t/deps_pch.json -DPREFIX=%/t

// CHECK: "-fmodule-format=obj"
// CHECK: "-dwarf-ext-refs"

// RUN: %deps-to-rsp %t/deps_pch.json --tu-index 0 > %t/pch.rsp
// RUN: %clang @%t/pch.rsp

// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-include-tree-full -cas-path %t/cas > %t/deps_tu.json
// RUN: FileCheck %s -input-file %t/deps_tu.json -DPREFIX=%/t

// RUN: %deps-to-rsp %t/deps_tu.json --tu-index 0 > %t/tu.rsp
// RUN: %clang @%t/tu.rsp

// RUN: cat %t/tu.ll | FileCheck %s -check-prefix=LLVMIR -DPREFIX=%/t
// LLVMIR: !DICompileUnit({{.*}}, splitDebugFilename: "prefix.pch"

// Extract include-tree casid
// RUN: cat %t/tu.rsp | sed -E 's|.*"-fcas-include-tree" "(llvmcas://[[:xdigit:]]+)".*|\1|' > %t/tu.casid

// RUN: clang-cas-test -cas %t/cas -print-include-tree @%t/tu.casid | FileCheck %s -check-prefix=INCLUDE_TREE -DPREFIX=%/t
// INCLUDE_TREE: (PCH) [[PREFIX]]/prefix.pch llvmcas://

// RUN: clang-scan-deps -compilation-database %t/cdb_no_preserve.json -format experimental-include-tree-full -cas-path %t/cas > %t/deps_no_preserve.json
// RUN: FileCheck %s -input-file %t/deps_no_preserve.json -DPREFIX=%/t -check-prefix=NO_PRESERVE

// Note: "raw" is the default format, so it will not show up in the arguments.
// NO_PRESERVE-NOT: "-fmodule-format=
// NO_PRESERVE-NOT: "-dwarf-ext-refs"


//--- cdb_pch.json.template
[{
  "directory": "DIR",
  "command": "clang -x c-header DIR/prefix.h -target x86_64-apple-macos12 -o DIR/prefix.pch -gmodules -g -Xclang -finclude-tree-preserve-pch-path",
  "file": "DIR/prefix.h"
}]

//--- cdb.json.template
[{
  "directory": "DIR",
  "command": "clang -S -emit-llvm DIR/tu.c -o DIR/tu.ll -include-pch DIR/prefix.pch -target x86_64-apple-macos12 -gmodules -g -Xclang -finclude-tree-preserve-pch-path",
  "file": "DIR/tu.c"
}]

//--- cdb_no_preserve.json.template
[{
  "directory": "DIR",
  "command": "clang -S -emit-llvm DIR/tu.c -o DIR/tu.ll -include-pch DIR/prefix.pch -target x86_64-apple-macos12 -gmodules -g",
  "file": "DIR/tu.c"
}]

//--- prefix.h
struct S {};

//--- tu.c
struct S s;
