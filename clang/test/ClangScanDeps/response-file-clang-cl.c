// Check that the scanner can adjust arguments by reading .rsp files in advance.

// RUN: rm -rf %t
// RUN: split-file %s %t

// First run the tests with a .cdb
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: sed -e "s|DIR|%/t|g" %t/args_nested.template > %t/args_nested.rsp

// RUN: cp %t/args_compilation.rsp %t/args.rsp
// RUN: clang-scan-deps --compilation-database %t/cdb.json > %t/deps.json
// RUN: cat %t/deps.json | sed 's:\\\\\?:/:g' | FileCheck -DPREFIX=%/t %s

// RUN: cp %t/args_preprocess.rsp %t/args.rsp
// RUN: clang-scan-deps --compilation-database %t/cdb.json > %t/deps.json
// RUN: cat %t/deps.json | sed 's:\\\\\?:/:g' | FileCheck -DPREFIX=%/t %s


// Now run the tests again with a in-place compilation database
// RUN: cd %t

// RUN: cp args_compilation.rsp args.rsp
// RUN: clang-scan-deps -o deps.json -- %clang_cl @args.rsp
// RUN: cat deps.json | sed 's:\\\\\?:/:g' | FileCheck -DPREFIX=%/t %s

// RUN: cp args_preprocess.rsp args.rsp
// RUN: clang-scan-deps -o deps.json -- %clang_cl @args.rsp
// RUN: cat deps.json | sed 's:\\\\\?:/:g' | FileCheck -DPREFIX=%/t %s

// Here we ensure that we got a qualified .obj with its full path, since that's what we're passing with /Fo
// CHECK: [[PREFIX]]/tu.obj:

//--- cdb.json.template
[{
  "file": "DIR/tu.cpp",
  "directory": "DIR",
  "command": "clang-cl @DIR/args.rsp"
}]

//--- args_compilation.rsp
@args_nested.rsp
/c

//--- args_preprocess.rsp
@args_nested.rsp
/E

//--- args_nested.template
/I include
tu.cpp
/FoDIR/tu.obj

//--- include/header.h

//--- tu.cpp
#include "header.h"
