// Check that the scanner can adjust arguments by reading .rsp files in advance.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: echo /Fo%t/tu.obj >> %t/args_nested.rsp

// RUN: echo /c >> %t/args_nested.rsp
// RUN: clang-scan-deps -compilation-database %t/cdb.json > %t/deps.json
// RUN: cat %t/deps.json | sed 's:\\\\\?:/:g' | FileCheck -DPREFIX=%/t %s

// RUN: echo /E >> %t/args_nested.rsp
// RUN: clang-scan-deps -compilation-database %t/cdb.json > %t/deps.json
// RUN: cat %t/deps.json | sed 's:\\\\\?:/:g' | FileCheck -DPREFIX=%/t %s

// Here we ensure that we got a qualified .obj with its full path, since that's what we're passing with /Fo
// CHECK: [[PREFIX]]/tu.obj:

//--- cdb.json.template
[{
  "file": "DIR/t.cpp",
  "directory": "DIR",
  "command": "clang-cl @DIR/args.rsp"
}]

//--- args.rsp
@args_nested.rsp
tu.cpp

//--- args_nested.rsp
/I include

//--- include/header.h

//--- tu.cpp
#include "header.h"
