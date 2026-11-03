// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.in > %t/cdb.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format \
// RUN:   experimental-full -optimize-args=all > %t/result.json
// RUN: cat %t/result.json | sed 's:\\\\\?:/:g' | FileCheck %s

// RUN: %deps-to-rsp %t/result.json --module-name=mod > %t/mod.rsp
// RUN: %clang @%t/mod.rsp -o %t/mod.pcm
// RUN: llvm-dwarfdump --debug-info %t/mod.pcm | FileCheck %s --check-prefix=DWARF
// DWARF: DW_TAG_compile_unit
// DWARF-NOT: DW_AT_comp_dir

//--- cdb.json.in
[{
  "directory": "DIR",
  "command": "clang -c -g -gmodules DIR/tu.c -fmodules -fmodules-cache-path=DIR/cache -IDIR/include/ -fdebug-compilation-dir=DIR -o DIR/tu.o",
  "file": "DIR/tu.c"
}]

//--- include/module.modulemap
module mod {
  header "mod.h"
}

//--- include/mod.h

//--- tu.c
#include "mod.h"

// Check the -fdebug-compilation-dir used for the module is the root
// directory when current working directory optimization is in effect.
// CHECK:  "modules": [
// CHECK: "command-line": [
// CHECK-NOT: -fdebug-compilation-dir
// CHECK:  "translation-units": [
